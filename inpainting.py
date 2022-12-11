#%%
from modules import UNet_conditional 
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from ddpm import Diffusion 
from utils import plot_images, get_data
import numpy as np
from PIL import Image
from argparse import Namespace

device = "cuda"
img_size = 64
nb_imgs = 6 

PATH = os.path.join("models","airplanes_foolish_lion5","ckpt12000.pt")
enable_cfg = True

args = Namespace(**{
    "dataset_path":"../datasets/airplane-dataset-trans",
    "img_size":img_size,
    "nb_imgs":nb_imgs,
    "device":"cuda",
    "batch_size":64,
    "classes_to_focus_on" : ['C-130_Hercules','B-29_Superfortress']
})


dl= get_data(args)
nb_labels = len(np.unique(np.array(dl.dataset.targets)))
                                                                                                    
diffusion = Diffusion(img_size=img_size, device=device)
model = UNet_conditional(image_size=img_size).to(device)
model.load_state_dict(torch.load(PATH))
model.eval()
# %%
from collections import Counter
dico = Counter(np.array(dl.dataset.targets))
idx_to_label = {v:k for k,v in dl.dataset.class_to_idx.items()}
label_count = {idx_to_label[idx]:val for idx,val in dico.items() }


# %% Create a mask
def create_squared_mask(coordinates, length, img_size):
    start_x, start_y = coordinates
    padding = length
    mask = torch.ones((img_size,img_size))
    mask[start_x:start_x+padding,start_y:start_y+padding]=0
    return mask
mask = create_squared_mask((15,15),38,img_size)
plt.imshow(mask)
# %%
im,label = dl.dataset[90]
print(im.shape)
print(label)
plt.imshow(im.permute(1,2,0))
plt.imshow(torch.mul(mask,im).permute(1,2,0))
# %%
import torchvision
img_p = "../datasets/airplane-dataset-trans/Airliner/0-24.jpg"
img= torchvision.transforms.functional.pil_to_tensor(Image.open(img_p))
img= torchvision.transforms.functional.resize(img, 64)
img= torchvision.transforms.functional.normalize(img/255,[0.5,0.5,0.5],[0.5,0.5,0.5])
plt.imshow(img.permute(1,2,0))
#%%
diffusion.resample_steps= 6
sampled_images = diffusion.inpaint(model, n=1, x_0=img.to(device), mask=mask.to(device))
# %%
plt.imshow(sampled_images.squeeze(0).permute(1,2,0).to("cpu"))
# %%
