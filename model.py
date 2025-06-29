import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from typing import List


# Time Embedding
class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps: int, embed_dim: int):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros(time_steps, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('embeddings', pe)

    def forward(self, t):
        return self.embeddings[t]  # [B, D]


# Attention
class Attention(nn.Module):
    def __init__(self, C, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (C // num_heads) ** -0.5
        self.qkv = nn.Linear(C, C * 3)
        self.proj = nn.Linear(C, C)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # [B, N, C]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v  
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out

def get_valid_group(num_channels, max_groups=8):
    for g in reversed(range(1, max_groups + 1)):
        if num_channels % g == 0:
            return g
    return 1

# ResBlock
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__()
        g1 = get_valid_group(in_c,8)
        g2 = get_valid_group(out_c,8)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.norm1 = nn.GroupNorm(g1,in_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.norm2 = nn.GroupNorm(g2,out_c)
        self.time_emb = nn.Linear(time_emb_dim, out_c)
        self.activation = nn.SiLU()
        self.residual_conv = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t_emb):
        h = self.activation(self.norm1(x))
        h = self.conv1(h)
        t = self.time_emb(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t
        h = self.activation(self.norm2(h))
        h = self.conv2(h)
        return h + self.residual_conv(x)

# DownBlock
class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim, with_attn=False):
        super().__init__()
        self.res1 = ResBlock(in_c, out_c, time_emb_dim)
        self.attn = Attention(out_c, num_heads=4) if with_attn else nn.Identity()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.attn(x)
        return self.pool(x), x  


# UpBlock
class UpBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c, time_emb_dim, with_attn=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.res1 = ResBlock(skip_c + out_c, out_c, time_emb_dim)
        self.attn = Attention(out_c, num_heads=4) if with_attn else nn.Identity()

    def forward(self, x, skip, t_emb):
        x = self.up(x)  # 上采样
        x = torch.cat([x, skip], dim=1)  # 跳跃连接
        x = self.res1(x, t_emb)
        x = self.attn(x)
        return x

# UNET
class UNET(nn.Module):
    def __init__(self, in_c=1, out_c=1, base_c=64, time_emb_dim=256, time_steps=1000,with_atten=True):
        super().__init__()
        self.time_embedding = nn.Sequential(
            SinusoidalEmbeddings(time_steps, time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        self.down1 = DownBlock(in_c, base_c, time_emb_dim, with_attn=True)
        self.down2 = DownBlock(base_c , base_c * 2, time_emb_dim, with_attn=True)
        self.down3 = DownBlock(base_c * 2, base_c * 4, time_emb_dim, with_attn=True)
        self.mid = ResBlock(base_c * 4, base_c * 8, time_emb_dim)
        self.up1 = UpBlock(base_c * 8, base_c * 4, base_c * 4, time_emb_dim, with_attn=True)
        self.up2 = UpBlock(base_c * 4, base_c * 2, base_c * 2, time_emb_dim, with_attn=True)
        self.up3 = UpBlock(base_c * 2, base_c, base_c, time_emb_dim, with_attn=True)
        self.out_conv= ResBlock(base_c, out_c,time_emb_dim)

    def forward(self, x, t):        
        t_emb = self.time_embedding(t)
        # encoder     
        x1, x1_copy = self.down1(x, t_emb)  
        x2, x2_copy = self.down2(x1, t_emb)  
        x3, x3_copy = self.down3(x2, t_emb)
        # Middle
        mid = self.mid(x3, t_emb)
        # Decoder
        x = self.up1(mid, x3_copy, t_emb)  
        x = self.up2(x, x2_copy, t_emb) 
        x = self.up3(x, x1_copy, t_emb) 
        return self.out_conv(x,t_emb)      