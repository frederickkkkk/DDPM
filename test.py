import torch
import torch.nn as nnny
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model import UNET
from diffusion import Diffusion
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNET(time_steps=1000).to(device)
model.load_state_dict(torch.load(r"E:\DDPM\DDPM_change\weights\epoch_87.pth"))
diffusion = Diffusion(32, 1000,device=device,cos_schedule=True)
tensor_to_pil = ToPILImage()
model.eval()
with torch.no_grad():
    x = diffusion.sample(model, 16,in_c = 1)
    x = (x + 1) / 2 
    grid = make_grid(x[:16], nrow=4)
    x_grid_pil = tensor_to_pil(grid)
    plt.imshow(x_grid_pil)
    plt.axis("off")
    plt.title("Sampled Images")
    plt.show()