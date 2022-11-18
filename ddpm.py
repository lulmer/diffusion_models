#%%
import os
import torch
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
        device="cuda",
    ):

        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def prepare_cosine_noise_schedule(self,s):
        t= torch.linspace(0, self.noise_steps-1, self.noise_steps)
        f_t = torch.cos((((t/self.noise_steps)+s)/(1+s))*(torch.pi/2))**2
        alpha_hat_t = f_t/f_t[0]
        beta = 1 - (alpha_hat_t[1:]/alpha_hat_t[:-1])
        return beta


    def noise_images(self, x: torch.Tensor, t: int) -> torch.Tensor:
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[
            :, None, None, None
        ]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon , epsilon

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
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = (
                    1
                    / torch.sqrt(alpha)
                    * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise)
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
