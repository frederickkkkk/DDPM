import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0] 
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)  


class Diffusion:
    def __init__(self, img_size, timesteps, device,cos_schedule=False,new_model=False):
        self.device = device
        self.timesteps = timesteps
        self.new_model=new_model
        if cos_schedule==True:
            self.beta = cosine_beta_schedule(timesteps).to(device)
        else:
            self.beta = torch.linspace(1e-4, 0.02, timesteps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
    def noise_images(self, x, t):
        noise = torch.randn_like(x)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(0, self.timesteps, (n,), device=self.device)
    
    def sample(self, model,num,in_c):
        model.eval()
        x = torch.randn((num, in_c, 32, 32)).to(self.device) 
        if self.new_model==False:
            for i in reversed(range(self.timesteps)):
                t = torch.full((num,), i, device=self.device, dtype=torch.long)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / alpha.sqrt() * (x - (1 - alpha) / (1 - alpha_hat).sqrt() * predicted_noise) + beta.sqrt() * noise
            return x
        else :
            for i in reversed(range(1, self.timesteps)): 
                t = torch.full((num,), i, device=self.device, dtype=torch.long)
                x0_pred = model(x, t) 
                t_minus_1 = torch.full((num,), i-1, device=self.device, dtype=torch.long)
                x ,_= self.noise_images(x0_pred, t_minus_1)  

            t = torch.zeros((num,), device=self.device, dtype=torch.long)
            x0 = model(x, t)
            return x0
 