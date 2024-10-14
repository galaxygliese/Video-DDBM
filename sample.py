#-*- coding:utf-8 -*-

from diffusion.unet3d import create_unet_model
from dit.dit import DiT_S_2, DiT_B_2
from diffusion import DdbmEdmDenoiser
from dataset import MovingMNIST
from diffusers.optimization import get_scheduler
from diffusers.models import AutoencoderKL
from torchvision.io import write_video
from torchvision import transforms as T

from typing import Optional
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch.nn as nn
import torch

parser = argparse.ArgumentParser()

# General options
parser.add_argument('--diffusion_timesteps', type=int, default=40) # Different from training
parser.add_argument('--warmup_steps', type=int, default=50)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=3)
parser.add_argument('-d', '--device', type=int, default=0)
parser.add_argument('-w', '--weight_path', type=str, default="./checkpoints/")
parser.add_argument('--sample_num', type=int, default=4)
parser.add_argument('--half', action='store_true')
parser.add_argument('--sw', action='store_true')
parser.add_argument('--dit', action='store_true')

# Dataset options
parser.add_argument('--dataset_path', type=str, default="./datas/val")
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--image_depth', type=int, default=10)
parser.add_argument('--in_channels', type=int, default=1)
parser.add_argument('--out_channels', type=int, default=1)

# Karras (EDM) options
parser.add_argument('--sigma_data', type=float, default=0.5)
parser.add_argument('--sigma_sample_density_mean', type=float, default=-1.2)
parser.add_argument('--sigma_sample_density_std', type=float, default=1.2)
parser.add_argument('--sigma_max', type=float, default=80)
parser.add_argument('--sigma_min', type=float, default=0.002)
parser.add_argument('--rho', type=float, default=7.0)
opt = parser.parse_args()

device = opt.device

@torch.no_grad()
def vae_encode(
        x:torch.Tensor, 
        vae:AutoencoderKL,
        VAE_CHANNEL:int = 4,
        VAE_DOWN_SAMPLE:int = 8
    ) -> torch.Tensor:
    B, C, T, H, W = x.shape
    if C == 1:
        C = 3
        x = x.repeat(1, 3, 1, 1, 1)
    x = x.permute(0, 2, 1, 3, 4) # (B, C, T, H, W) -> (B, T, C, H, W)
    x = x.reshape(B*T, C, H, W)
    x = vae.encode(x).latent_dist.sample().mul_(0.18215)
    x = x.reshape(B, T, VAE_CHANNEL, H//VAE_DOWN_SAMPLE, W//VAE_DOWN_SAMPLE)
    # x = x.reshape(B, T, VAE_CHANNEL, -1)
    x = x.permute(0, 2, 1, 3, 4) # (B, T, C, D) -> (B, C, T, D)
    return x

@torch.no_grad()
def vae_decode(
        z:torch.Tensor,
        vae:AutoencoderKL,
        input_channel:int = 3,
        input_height:int = 64,
        input_width:int = 64
    ) -> torch.Tensor:
    B, C, T, PH, PW = z.shape
    z = z.permute(0, 2, 1, 3, 4) # (B, C, T, H, W) -> (B, T, C, H, W)
    z = z.reshape(B*T, C, PH, PW)
    samples = vae.decode(z / 0.18215).sample
    x = samples.reshape(B, T, input_channel, input_height, input_width)
    x = x.permute(0, 2, 1, 3, 4)
    return x

@torch.no_grad()
def generate(
        model:DdbmEdmDenoiser, 
        y: torch.Tensor,
        num_diffusion_iters:int, 
        export_name:str, 
        sample_num:int,
        fps:int=5, 
        device:str='cuda',
        vae:Optional[AutoencoderKL] = None,
    ):
    # B = sample_num
    videos = None
    with torch.no_grad():
        if opt.dit:
            B, C, T, H, W = y.shape
            if C == 1:
                C = 3
            y = y.repeat(1, 3, 1, 1, 1)
        video_0 = 0.5*(y.permute(0, 2, 1, 3, 4).detach().to('cpu') + 1).clamp(0, 1)[0]
        video_0 = 255*video_0

        if not opt.dit:
            video_0 = video_0.repeat(1, 3, 1, 1) # for C=1
        video_0 = video_0.permute(0, 2, 3, 1).numpy()
        videos =  video_0 # 
        # initialize action from Guassian noise
        for i in range(sample_num):
            if opt.dit:
                z_y = vae_encode(y, vae)
                z, path = model.sample(z_y, steps=num_diffusion_iters)
                nimage = vae_decode(z, vae)
            else:
                nimage, path = model.sample(y, steps=num_diffusion_iters)
                
            y = nimage
                
            video = 0.5*(nimage.permute(0, 2, 1, 3, 4).detach().to('cpu') + 1).clamp(0, 1)[0] # (B, C, T, H, W) to (B, T, C, H, W)
            video = 255*video

            if not opt.dit:
                video = video.repeat(1, 3, 1, 1) # for C=1
            video = video.permute(0, 2, 3, 1).numpy()
            videos = np.concatenate([videos, video])
            print(">", videos.shape)
        
    write_video(export_name, videos, fps=5)
    
@torch.no_grad()
def generate_overlap(
        model:DdbmEdmDenoiser, 
        y: torch.Tensor,
        num_diffusion_iters:int, 
        export_name:str, 
        sample_num:int,
        fps:int=5, 
        depth:int=10,
        device:str='cuda',
        vae:Optional[AutoencoderKL] = None,
    ):
    with torch.no_grad():
        if opt.dit:
            B, C, T, H, W = y.shape
            if C == 1:
                C = 3
            y = y.repeat(1, 3, 1, 1, 1)
        video_0 = 0.5*(y.permute(0, 2, 1, 3, 4).detach().to('cpu') + 1).clamp(0, 1)[0]
        video_0 = 255*video_0
        video_0 = video_0.repeat(1, 3, 1, 1)
        video_0 = video_0.permute(0, 2, 3, 1).numpy()
        videos =  video_0 # 
        # initialize action from Guassian noise
        for i in range(sample_num):
            if opt.dit:
                z_y = vae_encode(y, vae)
                z, path = model.sample(z_y, steps=num_diffusion_iters)
            else:
            # y: (B, C, T, H, W) 
                nimage, path = model.sample(y, steps=num_diffusion_iters)
            
            half_y = y[:,:,depth//2:, ...]
            half_nimage = nimage[:,:,:depth//2, ...]
            y = torch.cat([half_y, half_nimage], dim=2)
            
            video = 0.5*(nimage.permute(0, 2, 1, 3, 4).detach().to('cpu') + 1).clamp(0, 1)[0] # (B, C, T, H, W) to (B, T, C, H, W)
            video = 255*video
            video = video.repeat(1, 3, 1, 1) # (T, C, H, W)
            video = video.permute(0, 2, 3, 1).numpy() # (T, H, W, C)
            videos = np.concatenate([videos, video[:depth//2, ...]])
            print(">", videos.shape)
        
    write_video(export_name, videos, fps=5)

def main():
    attention_type = 'flash' if opt.half else 'vanilla' 
    if opt.dit:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
        model = DiT_S_2(
            in_channels=4,
            input_size=(opt.image_depth, opt.image_size // 8, opt.image_size // 8),
            condition='label', # To ignore text_encoder
            learn_sigma=False,
        ).to(device) 
        print("DiT Model Loaded!")
    else:
        vae = None
        model = create_unet_model(
            image_size=opt.image_size,
            num_channels=opt.num_channels,
            num_res_blocks=opt.num_res_blocks,
            in_channels=opt.in_channels,
            use_fp16=opt.half,
            attention_type=attention_type
        ).to(device) 
        print("UNet Model Loaded!")
    model.load_state_dict(torch.load(opt.weight_path))
    model = DdbmEdmDenoiser(
        unet=model,
        sigma_data=opt.sigma_data,
        sigma_min=opt.sigma_min,
        sigma_max=opt.sigma_max,
        rho=opt.rho,
        device=device,
    )
    model.eval()
    print("Model Loaded!")
    print("Using Sliding Window Sampling:", opt.sw)
    
    transform = T.Compose([
        T.Lambda(lambda t: torch.tensor(t).float()),
        T.Lambda(lambda t: (t / 255. * 2) - 1), # img in [-1, 1] normalizing
        T.Lambda(lambda t: t.permute(1, 0, 2, 3)), # TCHW -> CTHW
    ])
    
    dataset = MovingMNIST(
        data_file=opt.dataset_path,
        # input_size=opt.image_size, 
        num_frames=opt.image_depth, 
        transform=transform
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True, 
        # don't kill worker process afte each epoch
        persistent_workers=True 
    )
    
    batch = next(iter(dataloader))
    print("batch.shape:", batch[0].shape, batch[1].shape)
    print("batch x range:", torch.max(batch[0]), torch.min(batch[0]))
    print("batch y range:", torch.max(batch[1]), torch.min(batch[1]))
    test_x, test_y = batch[0], batch[1]
    test_y = test_y.to(device)
    
    if not opt.sw:
        generate(
            model=model,
            y=test_y,
            num_diffusion_iters=opt.diffusion_timesteps,
            export_name=f"data/sample.mp4" if not opt.dit else f"data/dit_sample.mp4",
            sample_num=opt.sample_num,
            vae=vae
        )
    else:
        generate_overlap(
            model=model,
            y=test_y,
            num_diffusion_iters=opt.diffusion_timesteps,
            export_name=f"data/sample.mp4" if not opt.dit else f"data/dit_sample.mp4",
            sample_num=opt.sample_num,
            vae=vae
        )
    print('Done!')
    
if __name__ == '__main__':
    main()