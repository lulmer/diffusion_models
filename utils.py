#%%
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt 
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Subset 

import os 

from torch.utils.data import Dataset
import numpy as np
class CustomSubset(Dataset):
    r"""
    Subset of a dataset at specified labels.

    Arguments:
        dataset (Dataset): The whole Dataset
        labels (sequence): labels in the whole set selected for subset
    """
    def __init__(self, dataset, labels):
        indices = [idx for idx, target in enumerate(dataset.targets) if target in [dataset.class_to_idx[label] for label in labels]]
        self.class_to_idx = {key:val for key,val in dataset.class_to_idx.items() if key in labels}
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.targets = np.array(dataset.targets)[indices]

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)

def plot_images(images):
    plt.figure(figsize=(32,32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1)
    ], dim=-2).permute(1,2,0).cpu())
    plt.show()

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images,**kwargs)
    ndarr = grid.permute(1,2,0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def get_data(args):
    transforms =  torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.img_size),
        torchvision.transforms.RandomResizedCrop(args.img_size, scale=(0.8,1.0)),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    
  
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)

    if args.classes_to_focus_on is not None :
        dataset = CustomSubset(dataset, args.classes_to_focus_on)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models",run_name), exist_ok=True)
    os.makedirs(os.path.join("results",run_name), exist_ok=True)

# %%
