#-*- coding:utf-8 -*-

from diffusion.unet3d import create_unet_model
import argparse
import torch 
import numpy as np
import random

parser = argparse.ArgumentParser()

# General options
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('-b', '--batchsize', type=int, default=8)
parser.add_argument('--diffusion_timesteps', type=int, default=40) # Different from training
parser.add_argument('--ema_power', type=float, default=0.75)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--export_folder', type=str, default="./checkpoints")
parser.add_argument('--warmup_steps', type=int, default=50)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=3)
parser.add_argument('--save-per-epoch', type=int, default=50)
parser.add_argument('-d', '--device', type=int, default=0)
parser.add_argument('--ema-rate', type=float, default=0.9999)
parser.add_argument('--half', action='store_true')

# Dataset options
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--image_size', type=int, default=256)
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

# LATENT_DIM = opt.image_size // 8
torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)

def main():
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

if __name__ == '__main__':
    main()