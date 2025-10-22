import math
import torch
import torch.nn.functional as F
import argparse

import matplotlib.pyplot as plt
from PIL import Image

from utils.image_point_selector import PointSelector
from utils.eigenvals_utils import eigenval_modification
from utils.transforms import (
	make_transform,
	get_input_sizes,
	)

def main(args):
    REPO_DIR = "./dinov3"
    CHECKPOINT = "weights/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

    # DINOv3 ViT models pretrained on web images
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=CHECKPOINT)
    model = model.to(device)

    if args.svd:
        model = eigenval_modification(model, args.printing, args.threshold)

    # DINOv3 model parameters
    num_heads = model.num_heads
    embed_dim = model.embed_dim
    patch_size = model.patch_size
    n_storage_tokens = model.n_storage_tokens

    # Load image 
    image_path = './images/fruits.jpg' # <---- Change this to visualize  a different image
    img = Image.open(image_path).convert("RGB")
    w_orig, h_orig = img.size

    # Selects the points for self-attention visualization
    point_selector = PointSelector(img)
    point_selector.root.mainloop()

    selected_points = point_selector.get_points()
    point_selector.root.destroy() # Clean up the GUI window
    print(f"Retrieved {len(selected_points)} points: {selected_points}")

    # Image preprocessing
    NUM_TOKENS = 18000 # <---- Increasing this value increases the resolution of the feature map
    base_h, base_w = get_input_sizes(img, NUM_TOKENS)

    transform = make_transform(base_h, base_w, 16) # Create the transformations applied to the image
    print(f"DINOv3 uses an input image size of ({base_h*16}, {base_w*16})")

    # Scale selected points
    # Map the clicked (x, y) coordinates on the original image to token indices in the feature map
    point_token_indices = []
    for x, y in selected_points:
        # Convert scaled coordinates to token indices
        x_token = int(x * (base_w / w_orig))
        y_token = int(y * (base_h / h_orig))
        
        # Calculate the 1D index in the flattened token sequence
        token_idx = y_token * base_w + x_token + n_storage_tokens + 1
        point_token_indices.append(token_idx)

    # DINOv3 processing
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16, enabled=True):
        transform_img = transform(img)[None] # (1, 3, H, W)
        block_idx = -1 # using last layer
        qkv = []
        hooks = [
            model.blocks[block_idx].attn.qkv.register_forward_hook(
            lambda self, input, output: qkv.append(output)
            )
        ]
        outputs = model(transform_img.to(device))

        for hook in hooks:
            hook.remove()

        qkv = qkv[0]

        # The code below is from 
        # https://github.com/facebookresearch/dinov3/blob/adc254450203739c8149213a7a69d8d905b4fcfa/dinov3/models/vision_transformer.py#L224
        rope = [(base_h, base_w)]
        if model.rope_embed is not None:
            rope_sincos = [model.rope_embed(H=H, W=W) for H, W in rope]
        else:
            rope_sincos = [None for r in rope]

        # The code below is from 
        # https://github.com/facebookresearch/dinov3/blob/adc254450203739c8149213a7a69d8d905b4fcfa/dinov3/layers/attention.py#L106
        B, N, _ = qkv.shape
        C = embed_dim

        qkv = qkv.reshape(B, N, 3, num_heads, C // num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        scale_factor = 1 / math.sqrt(C)

        if rope_sincos[0] is not None:
            q, k = model.blocks[block_idx].attn.apply_rope(q, k, rope_sincos[0])

        # Compute attention weights
        attention_weights = F.softmax(q @ k.transpose(-1, -2) * scale_factor, dim=-1)
        attention_weights = attention_weights[0, :, point_token_indices].mean(dim=0)
        attention_weights = attention_weights[:, n_storage_tokens+1:]
        attention_weights = attention_weights.reshape(len(point_token_indices), base_h, base_w)
        
    # Visualization
    num_points = len(selected_points)
    fig, axes = plt.subplots(2, num_points, figsize=(8, 4 * num_points))

    # If only one point is selected, axes is not a 2D array, so we wrap it
    if num_points == 1:
        axes = axes[:, None]

    for i in range(num_points):
        # Plot original image with the selected point
        ax_img = axes[0][i]
        ax_img.imshow(img)
        ax_img.axis('off')
        
        # Overlay the selected point as a red '+'
        point_x, point_y = selected_points[i]
        ax_img.plot(point_x, point_y, 'r+', markersize=15, markeredgewidth=2)

        # Plot the attention heatmap
        ax_attn = axes[1][i]
        heatmap = attention_weights[i].cpu().numpy()
        im = ax_attn.imshow(heatmap, cmap='viridis')
        ax_attn.axis('off')

    plt.tight_layout()
    plt.suptitle(f"Attention map for Block {block_idx}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--svd", action="store_true",)
    parser.add_argument("-p", "--printing", action="store_true")
    parser.add_argument("-t", "--threshold", type=int, default=10)
    args = parser.parse_args()

    main(args)