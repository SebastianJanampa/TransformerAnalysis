import torch

REPO_DIR = "./dinov3"
CHECKPOINT = "weights/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

# DINOv3 ViT models pretrained on web images
device = 'cpu' 
model = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=CHECKPOINT)
model = model.to(device)

# Extract Wq, Wk, Wv for each transformer block
qkv_layer_weights = []
qkv_layer_bias = []

for block in model.blocks:
    qkv_layer_weights.append(block.attn.qkv.weight.detach())
    qkv_layer_bias.append(block.attn.qkv.bias.detach())

for weight, bias, in zip(qkv_layer_weights, qkv_layer_bias):
    print(weight.shape, bias.shape)