import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
from utils import setup_logging, get_data, save_images
from modules import UNet_conditional, EMA
from ddpm import Diffusion
import argparse
import copy
import numpy as np



def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    if args.cfg:
        nb_labels = len(np.unique(np.array(dataloader.dataset.targets)))
        model = UNet_conditional(image_size=args.img_size, num_classes=nb_labels).to(
            device
        )
    else:
        model = UNet_conditional(image_size=args.img_size).to(device)

    if args.resume_ckpt:
        print("Resuming training from a checkpoint")
        model.load_state_dict(torch.load(args.resume_ckpt))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3000,6000,10000], gamma=0.8)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    if args.ema:
        ema = EMA(beta=0.995)
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1 :
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if args.ema:
                ema.step_ema(ema_model, model)
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 1000 == 0:
            sampled_images = diffusion.sample(model, n=6)
            save_images(
                sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg")
            )
            

            if args.cfg:
                sampled_images = diffusion.sample(model, n=6,labels=torch.Tensor([0,0,0,2,2,2]).long())
                save_images(
                    sampled_images, os.path.join("results", args.run_name, f"guided_{epoch}.jpg")
                )
            torch.save(
                model.state_dict(),
                os.path.join("models", args.run_name, f"ckpt{epoch}.pt"),
            )
            if args.ema:
                '''
                ema_sampled_images = diffusion.sample(ema_model, n=6)
                save_images(
                    ema_sampled_images,
                    os.path.join("results", args.run_name, f"ema_{epoch}.jpg"),
                )
                '''
                torch.save(
                    model.state_dict(),
                    os.path.join("models", args.run_name, f"ema_ckpt{epoch}.pt"),
                )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a diffusion model")
    parser.add_argument("-epochs", default=12, type=int, help="number of epochs")
    parser.add_argument(
        "-batch_size", default=32, type=int, help="batch size of the model"
    )
    parser.add_argument(
        "-device", default="cuda", type=str, help="device you want to use for training"
    )
    parser.add_argument(
        "-img_size", type=int, default=128, help="image size processed by the model"
    )
    parser.add_argument(
        "-run_name", type=str, default="sleazy_donkey", help="name of the experiment"
    )
    parser.add_argument("-lr", type=float, default=3.0e-4, help="learning rate")
    parser.add_argument(
        "-dataset_path", type=str, default="dataset", help="dataset path"
    )
    parser.add_argument(
        "-scheduler",
        type=str,
        default="cosine",
        help="Noise scheduler can be either linear or cosine",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        default=False,
        help="wether apply Exponential moving average",
    )
    parser.add_argument(
        "--cfg",
        action="store_true",
        default=False,
        help="wether use Classifier Free Guidance",
    )
    parser.add_argument(
        "-resume_ckpt",
        type=str,
        default=None,
        help="checkpoint path to resume training",
    )
    parser.add_argument(
        "-classes_to_focus_on",
        nargs='+',
        default=None,
        help="List of classes_to_focus_on",
    )

    args = parser.parse_args()
    print(args)

    train(args)
