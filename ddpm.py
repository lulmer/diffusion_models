#%%
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt



logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


class Diffusion(object):
    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=1e-2,
        img_size=64,
        schedule_name="linear",
        device="cuda",
    ):

        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.schedule_name=schedule_name

        # define schedule
        self.betas = self.get_beta_schedule().to(device)
        # Define alphas 
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
   
    def get_beta_schedule(self):
        if self.schedule_name == "linear":
            return self.linear_noise_schedule()
        else :
            return self.cosine_noise_schedule()

    def linear_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def cosine_noise_schedule(self,s=0.008):
        t= torch.linspace(0, self.noise_steps-1, self.noise_steps)
        f_t = torch.cos((((t/self.noise_steps)+s)/(1+s))*(torch.pi/2))**2
        alphas_cumprod = f_t/f_t[0]
        betas = 1 - (alphas_cumprod[1:]/alphas_cumprod[:-1])
        return torch.clip(betas,0.0001,0.9999)


    def noise_images(self, x: torch.Tensor, t: int) -> torch.Tensor:
        epsilon = torch.randn_like(x)
        return torch.sqrt(self.alphas_cumprod[t])[:,None,None,None] * x + torch.sqrt(1 - self.alphas_cumprod[t])[:,None,None,None] * epsilon , epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn(n, 3, self.img_size, self.img_size).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alphas[t][:,None,None,None]
                alpha_cumprod = self.alphas_cumprod[t][:,None,None,None]
                beta = self.betas[t][:,None,None,None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = (
                    1
                    / torch.sqrt(alpha)
                    * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise)
                    + torch.sqrt(beta) * noise
                )
                '''
                x_aux = (x.clamp(-1, 1) + 1) / 2
                x_aux = (x_aux * 255).type(torch.uint8)
                x_aux = x_aux.squeeze(0)
                x_aux = x_aux.permute(1,2,0)
                plt.imshow(x_aux.to('cpu'))
                plt.show()
                '''
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        
        return x
    




# %%
