#%% 
from modules import UNet_conditional 
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from ddpm import Diffusion 
from utils import plot_images, get_data
import numpy as np
from argparse import Namespace
device = "cuda"
img_size = 64
nb_imgs = 6 
PATH = os.path.join("models","naughty_elephant","ckpt300.pt")
enable_cfg = True
#%%
args = {
    "dataset_path":"../datasets/eurosat",
    "img_size":64,
    "nb_imgs":6,
    "device":"cuda",
    "batch_size":64,
    "classes_to_focus_on" : None
}
args = Namespace(**args)
dl= get_data(args)
nb_labels = len(np.unique(np.array(dl.dataset.targets)))
diffusion = Diffusion(img_size=img_size, device=device)
model = UNet_conditional(image_size=img_size, num_classes=nb_labels).to(device)
model.load_state_dict(torch.load(PATH))
model.eval()
sampled_images = diffusion.sample(model, n=nb_imgs, labels=torch.Tensor([0,5,4,3,2,1]).long())

plot_images(sampled_images)

#%%
args = Namespace(**{
    "dataset_path":"../datasets/airplane-dataset-trans",
    "img_size":64,
    "nb_imgs":6,
    "device":"cuda",
    "batch_size":64,
    "classes_to_focus_on" : ['C-130_Hercules']
})

dl= get_data(args)
nb_labels = len(np.unique(np.array(dl.dataset.targets)))
#%%
from collections import Counter
dico = Counter(np.array(dl.dataset.targets))
idx_to_label = {v:k for k,v in dl.dataset.class_to_idx.items()}
label_count = {idx_to_label[idx]:val for idx,val in dico.items() }

# %%
sampled_images = diffusion.sample(model, n=nb_imgs)
# %%
import matplotlib.pyplot as plt 

plt.barh(list(label_count.keys()),list(label_count.values()), color ='maroon')
# %%
dl.dataset[6510][1]