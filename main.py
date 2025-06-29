import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model import UNET
from diffusion import Diffusion
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import os

plt.ion()
tensor_to_pil = ToPILImage()

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x - 0.5) * 2)
])

modelConfig = {
    "epoch": 100,
    "batch_size": 128,
    "lr": 1e-4,
    "img_size": 32,
    "in_c":1,
    "timesteps":300,
    "data":'./MNIST',
    "with_atten":True,
    "cos_schedule":False,
    "new_model":False,
}
dataset = datasets.MNIST(modelConfig["data"], train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, modelConfig["batch_size"], shuffle=True, pin_memory=True)

model = UNET(in_c=modelConfig["in_c"],out_c=modelConfig["in_c"],
             time_steps=modelConfig["timesteps"],with_atten=modelConfig["with_atten"]).to(device)
optimizer = torch.optim.Adam(model.parameters(), modelConfig["lr"])
diffusion = Diffusion(modelConfig["img_size"], modelConfig["timesteps"],device=device, 
            cos_schedule=modelConfig["cos_schedule"],new_model=modelConfig["new_model"])
scaler = torch.cuda.amp.GradScaler()

epochs = modelConfig["epoch"]
for epoch in range(epochs):
    with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)

            with torch.cuda.amp.autocast():
                if modelConfig["new_model"]==False:
                    predicted_noise = model(x_t, t)
                    loss = nn.functional.mse_loss(noise, predicted_noise)
                else:
                    predicted_x = model(x_t, t)
                    loss = nn.functional.mse_loss(images, predicted_x)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(loss=loss.item())

    torch.cuda.empty_cache()

    model.eval()
    with torch.no_grad():
        x = diffusion.sample(model, 16,modelConfig["in_c"])
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        grid = make_grid(x[:16], nrow=4)
        pil_img = tensor_to_pil(grid)

        save_path = r"E:\DDPM\DDPM_change\result\epoch_{}.png".format(epoch+1)
        pil_img.save(save_path)

    # 保存模型权重
    if epoch % 3 == 0:
        os.makedirs(r"E:/DDPM/DDPM_change/weights", exist_ok=True)
        weight_path = f"E:/DDPM/DDPM_change/weights/epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), weight_path)
