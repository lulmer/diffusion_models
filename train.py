import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
from utils import setup_logging, get_data, save_images
from modules import UNet
from ddpm import Diffusion
import argparse

parser = argparse.ArgumentParser(description='Train a diffusion model')
parser.add_argument('epochs', default=12, type=int,
                    help='number of epochs')
parser.add_argument('device', default='cuda', type=str,
                    help='device you want to use for training')                    
parser.add_argument('img_size', type=int, default=250,
                    help='image size processed by the model')
parser.add_argument('run_name', type=str, default='sleazy_donkey',
                    help='name of the experiment')
parser.add_argument('lr', type=float, default=10E-3,
                    help='learning rate')

args = parser.parse_args()


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr = args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l= len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images,t)
            
            predicted_noise = model(x_t,t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch*l+i)
        
        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, os.path.join("results",args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models",args.run_name, f"ckpt.pt"))