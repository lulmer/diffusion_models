#%%
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import numpy as np 

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
        resample_steps = 25,
        schedule_name="linear",
        device="cuda",
    ):

        self.noise_steps = noise_steps
        self.resample_steps = resample_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.num_inference_steps = 100
        self.eta = 0
        self.device = device
        self.schedule_name = schedule_name

        # define schedule
        self.betas = self.get_beta_schedule().to(device)
        # Define alphas
        self.alphas = (1 - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(device)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).to(device)

    def get_beta_schedule(self):
        if self.schedule_name == "linear":
            return self.linear_noise_schedule()
        elif self.schedule_name == "cosine":
            return self.cosine_noise_schedule()
        elif self.schedule_name == "quadratic":
            return self.quadratic_beta_schedule()
        elif self.schedule_name == "sigmoid":
            return self.sigmoid_beta_schedule()

    def linear_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def cosine_noise_schedule(self, s=0.008):
        steps= self.noise_steps+1
        t = torch.linspace(0, self.noise_steps, steps)
        f_t = torch.cos((((t / self.noise_steps) + s) / (1 + s)) * torch.pi * 0.5) ** 2
        alphas_cumprod = f_t / f_t[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def quadratic_beta_schedule(self):
        return torch.linspace(self.beta_start**0.5, self.beta_end**0.5, self.noise_steps) ** 2

    def sigmoid_beta_schedule(self):
        betas = torch.linspace(-6, 6, self.noise_steps)
        return torch.sigmoid(betas) * (self.beta_end - self.beta_start) + self.beta_start

    def noise_images(self, x: torch.Tensor, t: int) -> torch.Tensor:
        epsilon = torch.randn_like(x)
        return (
            torch.sqrt(self.alphas_cumprod[t])[:, None, None, None] * x
            + torch.sqrt(1 - self.alphas_cumprod[t])[:, None, None, None] * epsilon,
            epsilon,
        )

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels=None, cfg_scale=3):
        
        logging.info(f"Sampling {n} new images....")
        model.eval()
        if labels is not None:
                    labels = labels.to(self.device)
        with torch.no_grad():
            x = torch.randn(n, 3, self.img_size, self.img_size).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n).to(self.device) * i).long().to(self.device)

                predicted_noise = model(x, t, labels)
                if cfg_scale > 0  :
                    uncond_predicted_noise = model(x,t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

                alpha = self.alphas[t][:, None, None, None]
                alpha_cumprod = self.alphas_cumprod[t][:, None, None, None]
                beta = self.betas[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = (
                    1
                    / torch.sqrt(alpha)
                    * (
                        x
                        - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod))
                        * predicted_noise
                    )
                    + torch.sqrt(beta) * noise
                )
                 
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)

        return x

    def sample_ddim(self, model, n, labels=None, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        if labels is not None:
                    labels = labels.to(self.device)
        with torch.no_grad():
            x = torch.randn(n, 3, self.img_size, self.img_size).to(self.device)
            step_ratio = self.noise_steps//self.num_inference_steps
            timesteps = (np.arange(0, self.num_inference_steps) * step_ratio).round().copy().astype(np.int64)
            self.timesteps = torch.from_numpy(timesteps).to(self.device)
            for i in tqdm(reversed(self.timesteps)):
                t = (torch.ones(n).to(self.device) * i).long().to(self.device)
                t_prev = t - step_ratio

                predicted_noise = model(x, t, labels)

                alpha_cumprod = self.alphas_cumprod[t][:, None, None, None]
                alpha_cumprod_prev = self.alphas_cumprod[t_prev][:, None, None, None] if t_prev.any() >= 0 else torch.Tensor(1.0)[:, None, None, None]

                if cfg_scale > 0  :
                    uncond_predicted_noise = model(x,t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
              
                # Compute "predicted x_0" as seen in formula (12) of the DDIM paper
                predicted_x_0 = (x - (torch.sqrt(1-alpha_cumprod) * predicted_noise))/torch.sqrt(alpha_cumprod)
                predicted_x_0 = torch.clamp(predicted_x_0, -1, 1)
                # Compute the variance as seen in formula (12) of the DDIM paper
                sigma_t = torch.sqrt((1-alpha_cumprod_prev)/(1-alpha_cumprod))*torch.sqrt(1- alpha_cumprod/alpha_cumprod_prev)
                std_dev_t = self.eta * torch.sqrt(sigma_t)  if self.eta > 0 else 0
                # Compute the "direction pointing to x_t" as seen in formula (12) of the DDIM paper
                direction_x_t=torch.sqrt(1-alpha_cumprod_prev-std_dev_t**2)* predicted_noise 

                x = torch.sqrt(alpha_cumprod_prev) * predicted_x_0 + direction_x_t +std_dev_t * noise

               
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)

        return x

    def inpaint(self, model, x_0, mask, n, labels=None, cfg_scale=3):
        logging.info(f"Inpainting {n} new images....")
        model.eval()
        if labels is not None:
                    labels = labels.to(self.device)
        with torch.no_grad():
            x = torch.randn(n, 3, self.img_size, self.img_size).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                for u in range(0, self.resample_steps):
                    t = (torch.ones(n) * i).long().to(self.device)

                    predicted_noise = model(x, t, labels)

                    if cfg_scale > 0  :
                        uncond_predicted_noise = model(x,t, None)
                        predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

                    alpha = self.alphas[t][:, None, None, None]
                    alpha_cumprod = self.alphas_cumprod[t][:, None, None, None]
                    beta = self.betas[t][:, None, None, None]
                    if i > 1:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)

                    x_known, _ =  self.noise_images(x_0,t)

                    x_unknown = (1/torch.sqrt(alpha))* (x- ((1 - alpha) / torch.sqrt(1 - alpha_cumprod))* predicted_noise)+ torch.sqrt(beta) * noise
                    

                    x = torch.mul(mask,x_known) + torch.mul(torch.ones_like(mask)-mask,x_unknown) 
                    if u < self.resample_steps and i>1 :
                        x = torch.sqrt(alpha) * x + noise * torch.sqrt(beta) 
    
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)

        return x


# %%
