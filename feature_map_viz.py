import math
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from PIL import Image

from utils.transforms import (
    do_pca,
    make_transform,
    get_input_sizes,
    )


REPO_DIR = "./dinov3"
CHECKPOINT = "weights/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

# DINOv3 ViT models pretrained on web images
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=CHECKPOINT)
model = model.to(device)

# DINOv3 model parameters
num_heads = model.num_heads
embed_dim = model.embed_dim
patch_size = model.patch_size
n_storage_tokens = model.n_storage_tokens

# Load image 
image_path = './images/fruits.jpg' # <---- Change this to visualize  a different image
img = Image.open(image_path).convert("RGB")
w_orig, h_orig = img.size

# Image preprocessing
NUM_TOKENS = 18000 # <---- Increasing this value increases the resolution of the feature map
base_h, base_w = get_input_sizes(img, NUM_TOKENS)

transform = make_transform(base_h, base_w, 16) # Create the transformations applied to the image
print(f"DINOv3 uses an input image size of ({base_h*16}, {base_w*16})")

# DINOv3 processing
with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16, enabled=True):
    transform_img = transform(img)[None] # (1, 3, H, W)
    feats = model.get_intermediate_layers(transform_img.to(device), n=[11], reshape=True, norm=True)

# Plot figures
fig = plt.figure(figsize=(24, 36))
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.axis('off')


# PCA transformation 
feats = feats[-1]
feat = feats[0]
x = feat.detach().cpu().float()
pca = do_pca(x)

plt.subplot(1, 3, 2)
plt.imshow(pca, cmap='viridis')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(x[0], cmap='viridis')
plt.axis('off')

plt.show()