#-*- coding:utf-8 -*-

from diffusion.fp16_util import MixedPrecisionTrainer
from diffusers.optimization import get_scheduler
from diffusion import DdbmEdmDenoiser
from diffusion.unet3d import create_unet_model
from dataset import MovingMNIST

from torch_ema import ExponentialMovingAverage
from torchvision import transforms as T
from torchvision.io import write_video
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from functools import partial
from PIL import Image
from tqdm.auto import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
import dill as pickle
import nibabel as nib
import torchio as tio
import argparse
import torch 
import numpy as np
import random
import wandb
import copy
import os

parser = argparse.ArgumentParser()

# General options
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('-b', '--batchsize', type=int, default=8)
parser.add_argument('--generate-batchsize', type=int, default=1)
parser.add_argument('--diffusion_timesteps', type=int, default=40) # Different from training -> Need to change?
parser.add_argument('--lr', type=float, default=1e-4) # -> Need to change?
parser.add_argument('--export_folder', type=str, default="./checkpoints")
parser.add_argument('--warmup_steps', type=int, default=50) # -> Need to change?
parser.add_argument('--num_channels', type=int, default=64) # -> Need to change?
parser.add_argument('--num_res_blocks', type=int, default=3) # -> Need to change?
parser.add_argument('--save-per-epoch', type=int, default=100)
parser.add_argument('--ema-rate', type=float, default=0.99) # -> Need to change?
parser.add_argument('--half', action='store_true')

# Dataset options
parser.add_argument('--data_path', type=str)
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--image_depth', type=int, default=10)
parser.add_argument('--in_channels', type=int, default=1)
parser.add_argument('--out_channels', type=int, default=1)

# Karras (EDM) options
parser.add_argument('--sigma_data', type=float, default=0.5)
parser.add_argument('--sigma_sample_density_mean', type=float, default=-1.2)
parser.add_argument('--sigma_sample_density_std', type=float, default=1.2)
parser.add_argument('--sigma_max', type=float, default=80)
parser.add_argument('--sigma_min', type=float, default=0.0002)
parser.add_argument('--rho', type=float, default=7.0)

# Resuming options
parser.add_argument('--resume', action='store_true')
parser.add_argument('--resume_checkpoint', type=str)
parser.add_argument('--resume_epochs', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
opt = parser.parse_args()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# LATENT_DIM = opt.image_size // 8
seed_everything(opt.seed)

def rand_log_normal(shape, loc=0., scale=1., device='cuda', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    return (torch.randn(shape, device=device, dtype=dtype) * scale + loc).exp()

@torch.no_grad()
def generate(
        model:DdbmEdmDenoiser, 
        y: torch.Tensor,
        num_diffusion_iters:int, 
        export_name:str, 
        # sample_num:int, 
        device:str='cuda'
    ):
    # B = sample_num
    with torch.no_grad():
        # initialize action from Guassian noise
        nimage, path = model.sample(y, steps=num_diffusion_iters)
        
        nimage = ((nimage + 1) * 0.5).clamp(0, 1) # (B, C, D, H, W)
        
        video = 255*nimage.permute(0, 2, 1, 3, 4).detach().to('cpu')[0]
        video = video.repeat(1, 3, 1, 1)
        video = video.permute(0, 2, 3, 1)
    write_video(export_name, video, fps=5)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def update_ema(mp_trainer:MixedPrecisionTrainer, ema_params, ema_rate:float=0.99):
    for targ, src in zip(ema_params, mp_trainer.master_params):
        targ.detach().mul_(ema_rate).add_(src, alpha=1 - ema_rate)

def anneal_lr(step:int, optimizer, lr_anneal_steps:float = 0):
    if lr_anneal_steps == 0:
        return
    frac_done = (step) / lr_anneal_steps
    lr = opt.lr * (1 - frac_done)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def debug():
    device = 'cuda'
    attention_type = 'flash' if opt.half else 'vanilla' 
    unet = create_unet_model(
        image_size=opt.image_size,
        num_channels=opt.num_channels,
        num_res_blocks=opt.num_res_blocks,
        in_channels=opt.in_channels,
        use_fp16=opt.half,
        attention_type=attention_type
    ).to(device) 
    print("Model Loaded!")

    x = torch.randn(1, 1, 64, 64, 64).to(device)
    y = torch.randn(1, 1, 64, 64, 64).to(device)
    t = torch.tensor([80.0]).to(device)
    o = unet(x, t, xT=y)
    print(">", o.shape)

def train(rank, world_size, run):
    setup(rank, world_size)
    device = rank
    step = 0
    
    transform = T.Compose([
        T.Lambda(lambda t: torch.tensor(t).float()),
        T.Lambda(lambda t: (t / 255. * 2) - 1), # img in [-1, 1] normalizing
        T.Lambda(lambda t: t.permute(1, 0, 2, 3)), # TCHW -> CTHW
    ])
    
    dataset = MovingMNIST(
        data_file=opt.data_path, 
        # input_size=opt.image_size, 
        num_frames=opt.image_depth, 
        transform=transform
    )
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=opt.seed
    )
    
    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(opt.batchsize // world_size),
        shuffle=False,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    num_diffusion_iters = opt.diffusion_timesteps   
    attention_type = 'flash' if opt.half else 'vanilla' 
    unet = create_unet_model(
        image_size=opt.image_size,
        num_channels=opt.num_channels,
        num_res_blocks=opt.num_res_blocks,
        in_channels=opt.in_channels,
        use_fp16=opt.half,
        attention_type=attention_type
    ).to(device)
    print("UNET dtype:", unet.dtype)
    if opt.half:
        mp_trainer = MixedPrecisionTrainer(
            model=unet,
            use_fp16=True,
            fp16_scale_growth=1e-3,
        )
    ddp_unet = DDP(
        unet, 
        device_ids=[rank],
        output_device=rank,
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )
    model = DdbmEdmDenoiser(
        unet=ddp_unet,
        sigma_data=opt.sigma_data,
        sigma_min=opt.sigma_min,
        sigma_max=opt.sigma_max,
        rho=opt.rho,
        device=device,
    )
    model.to(device)
    
    if opt.half:
        ema_params = copy.deepcopy(mp_trainer.master_params)
    print("Model Loaded!")
    
    if opt.resume:
        state_dict = torch.load(opt.resume_checkpoint, map_location='cuda')
        model.load_state_dict(state_dict)
        
        # TODO: load EMA params
        print("Pretrained Model Loaded")
            
    if rank == 0:
        batch = dataset[0]
        print("batch.shape:", batch[0].shape, batch[1].shape)
        print("batch x range:", torch.max(batch[0]), torch.min(batch[0]))
        print("batch y range:", torch.max(batch[1]), torch.min(batch[1]))
        test_y = torch.cat([batch[1].unsqueeze(0)]*opt.generate_batchsize, 0).to(device)
        print("test y:", test_y.shape)
    
    if opt.half:
        optimizer = torch.optim.AdamW(
            params=mp_trainer.master_params, 
            lr=opt.lr, weight_decay=1e-6
        )
    else:
        optimizer = torch.optim.AdamW(
            params=ddp_unet.parameters(), 
            lr=opt.lr, weight_decay=1e-6
        )
    
    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=opt.warmup_steps,
        num_training_steps=len(dataloader) * opt.epochs
    )

    ema = ExponentialMovingAverage(model.parameters(), decay=opt.ema_rate)
    
    with tqdm(range(opt.resume_epochs, opt.epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            model.unet.train()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    x, y = nbatch[0].to(device), nbatch[1].to(device)
                    B = x.shape[0]
                    
                    # sample a diffusion iteration for each data point
                    loss = model.get_loss(x_start=x, x_T=y).mean()

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
        
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()
                    
                    # if opt.half:
                    #     took_step = mp_trainer.optimize(optimizer)
                    #     if took_step:
                    #         update_ema(mp_trainer, ema_params, ema_rate=opt.ema_rate)
                    # anneal_lr(step, optimizer)
                    step += 1
                    ema.update()

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
                if rank == 0:
                    run.log({"train-loss": np.mean(epoch_loss)})
                tglobal.set_postfix(loss=np.mean(epoch_loss))   
            
                if (epoch_idx + 1) % opt.save_per_epoch == 0 and rank == 0:
                    torch.save(model.unet.module.state_dict(), f'{opt.export_folder}/training-3dddbm-edm-latent-diffusion-epoch{epoch_idx+1}'+'.pt')
                    model.unet.eval()
                    with ema.average_parameters():
                        generate(
                            model=model,
                            y=test_y,
                            num_diffusion_iters=num_diffusion_iters,
                            export_name=f"{opt.export_folder}/epoch{epoch_idx+1}.mp4",
                            # sample_num=4
                        )
    torch.save(model.unet.module.state_dict(), f'{opt.export_folder}/final-3dddbm-edm-latent-diffusion-epoch{epoch_idx+1}'+'.pt')
    model.unet.eval()
    generate(
        model=model,
        y=test_y,
        num_diffusion_iters=num_diffusion_iters,
        export_name=f"{opt.export_folder}/epoch{epoch_idx+1}.nii.gz",
    )
    cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    
    # TODO: fix wandb resume
    run = wandb.init(project = 'video_ddbm', resume = opt.resume)
    config = run.config
    config.epochs = opt.epochs
    config.batchsize = opt.batchsize
    config.learning_rate = opt.lr 
    config.diffusion_timesteps = opt.diffusion_timesteps

    config.image_size = opt.image_size
    config.image_depth = opt.image_depth

    config.sigma_data = opt.sigma_data
    config.sigma_sample_density_mean = opt.sigma_sample_density_mean
    config.sigma_sample_density_std = opt.sigma_sample_density_std
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    mp.spawn(train, args=(world_size, run), nprocs=world_size, join=True)