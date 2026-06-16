import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np


# ==== Parameters ====
image_size = 224      # Input image: 224 x 224
patch_size = 16       # Each patch: 16 x 16
num_patches = (image_size // patch_size) ** 2  # 14x14 = 196 patches
d_model = 768         # Embedding size
batch_size = 1


# ==== Dummy Input (random image) ====
# Or replace with real image: image = T.ToTensor()(Image.open('your_image.jpg').resize((224, 224)))
image = torch.randn(batch_size, 3, image_size, image_size)  # (B, C, H, W)


# ==== Step 1: Convert image into non-overlapping patches ====
def extract_patches(img, patch_size):
   B, C, H, W = img.shape
   assert H % patch_size == 0 and W % patch_size == 0
   patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
   # Result: (B, C, num_patches_H, num_patches_W, patch_size, patch_size)
   patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
   patches = patches.permute(0, 2, 1, 3, 4)  # (B, N, C, Ph, Pw)
   patches = patches.reshape(B, -1, patch_size*patch_size*3)  # (B, N, 768)
   return patches


patches = extract_patches(image, patch_size)  # (B, 196, 768)


# ==== Step 2: Linear projection to embedding space ====
patch_embedding = nn.Linear(patch_size*patch_size*3, d_model)  # (768 -> 768)
x = patch_embedding(patches)  # (B, 196, 768)


# ==== Step 3: Add CLS token ====
cls_token = nn.Parameter(torch.zeros(1, 1, d_model))  # (1, 1, 768)
cls_tokens = cls_token.expand(batch_size, -1, -1)     # (B, 1, 768)
x = torch.cat([cls_tokens, x], dim=1)                 # (B, 197, 768)


# ==== Step 4: Add positional embeddings ====
pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))  # (1, 197, 768)
x = x + pos_embed  # Element-wise addition (broadcast across batch)


# ==== Final Output ====
print("Final x shape:", x.shape)  # (B, 197, 768)
