#%%
from modules import UNet
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from ddpm import Diffusion 
from utils import plot_images, get_data

device = "cuda"
img_size = 64
nb_imgs = 6 
PATH = os.path.join("models","sleazy_donkey","ckpt4.pt")

diffusion = Diffusion(img_size=img_size, device=device)
model = UNet(image_size=img_size).to(device)
model.load_state_dict(torch.load(PATH))
model.eval()
sampled_images = diffusion.sample(model, n=nb_imgs)

plot_images(sampled_images)

#%%
from utils import plot_images, get_data
from argparse import Namespace
args = {
    "dataset_path":"../datasets/dog_dataset/images",
    "img_size":128,
    "nb_imgs":6,
    "device":"cuda",
    "batch_size":64
}
args = Namespace(**args)
dl= get_data(args)
dl.dataset[0]

# %%
sampled_images = diffusion.sample(model, n=nb_imgs)
# %%
