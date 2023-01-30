#%% 
from modules import UNet_conditional 
import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
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
import albumentations as A  
#%%
args = {
    "dataset_path":"../datasets/eurosat",
    "img_size":64,
    "nb_imgs":6,
    "device":device,
    "batch_size":64,
    "classes_to_focus_on" : None
}
args = Namespace(**args)
dl= get_data(args)
nb_labels = len(np.unique(np.array(dl.dataset.targets)))
diffusion = Diffusion(img_size=img_size, device=device, schedule_name="linear")
diffusion.num_inference_steps=100
model = UNet_conditional(image_size=img_size, num_classes=nb_labels, device=device).to(device)
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
import torchvision
from PIL import Image
img_p = "../datasets/airplane-dataset-trans/Airliner/0-24.jpg"
img= torchvision.transforms.functional.pil_to_tensor(Image.open(img_p))
img= torchvision.transforms.functional.resize(img, 64)
img= torchvision.transforms.functional.normalize(img/255,[0.5,0.5,0.5],[0.5,0.5,0.5])
plt.imshow(img.permute(1,2,0))
img = img.to(device)
noise_schedule_types = ['linear', 'cosine', 'quadratic', 'sigmoid']
diffusion = Diffusion(img_size=img_size, device=device, schedule_name=noise_schedule_types[3])
#img = dl.dataset[610][0].to(device)
timesteps = torch.linspace(0,999,8).long().to(device)
for i,t in enumerate(timesteps):
    x_noised,_  = diffusion.noise_images(img.unsqueeze(0), t.unsqueeze(0))
    if i == 0:
        images = x_noised
    else : 
        images = torch.cat((images,x_noised),0)
    
plot_images(images)
    








# %%
import torchvision
from torch.utils.data import DataLoader
import cv2


transforms = A.Compose([
    A.Resize(width=args.img_size, height=args.img_size),
    A.Rotate(limit=10, p=0.3,border_mode=cv2.BORDER_REPLICATE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    A.pytorch.transforms.ToTensorV2(),
    ]
)

class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))

transforms =  torchvision.transforms.Compose([
    torchvision.transforms.Resize(args.img_size),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

plt.imshow(dl.dataset[604][0].permute(1,2,0))
# %%
plt.imshow(dl.dataset[600][0].permute(1,2,0))
# %%
