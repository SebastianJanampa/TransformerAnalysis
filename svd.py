import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import StrMethodFormatter, PercentFormatter
from matplotlib.lines import Line2D

import math

REPO_DIR = "./dinov3"
CHECKPOINT = "weights/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

device = 'cpu'
model = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=CHECKPOINT)
model = model.to(device)

num_heads = model.num_heads
embed_dim = model.embed_dim

qkv_layer_weights = []
proj_layer_weights = []
for block in model.blocks:
    weights = block.attn.qkv.weight.detach() # in_feats, out_feats
    weights = weights.view(num_heads, 3, embed_dim // num_heads, embed_dim)
    qkv_layer_weights.append(weights)

    weights = block.attn.proj.weight.detach()
    proj_layer_weights.append(weights)

def thresholding(eigenvals,  thresholding):
    max_sigma = eigenvals[0]
    n = len(eigenvals) - 1
    cond_num = max_sigma / eigenvals[n]
    while n > 4 and cond_num > thresholding:
        n-=1
        cond_num= max_sigma / eigenvals[n]
    eigenvals = eigenvals[:n+1]
    return eigenvals, n+1

def extract_spaces(A):
    rank = np.linalg.matrix_rank(A)
    U, S, Vh = np.linalg.svd(A)
    S = S
    row_space = Vh[:rank, :].T
    null_space = Vh[rank: :].T
    col_space = U[:, :rank]
    left_null = U[:, :rank]

    return [row_space, null_space, col_space, left_null], [U, S, Vh], rank

PRINT=True

# For Wq, Wk, Wv
SPACES, SVD = [], []
max_n = 0
for block_id, layer_weights in enumerate(qkv_layer_weights, start=1):
    block = layer_weights[:, 1, :, :].cpu().numpy() # (n_heads, in_feats, out_feats)

    if PRINT:
        print("#"*36 + f"Transformer Block {block_id:02d}" + "#"*37)

    spaces, svd = [], []
    for head_id, head  in enumerate(block, start=1):
        space_head, svd_head, rank = extract_spaces(head)
        svd_head[1], n = thresholding(svd_head[1], 100)
        max_n = max(max_n, n)
        spaces.append(space_head), svd.append(svd_head)

        max_sigma = svd_head[1].max()
        min_sigma = svd_head[1].min()
        if PRINT:
            print("="*36 + f"Head {head_id:02d}" + "="*37)
            print(f"rank:{rank}     new num of σs: {n}  max σ: {max_sigma:.3f}   min σ: {min_sigma:.3f}   (max σ / min σ): {max_sigma/min_sigma:.3f}")

    SPACES.append(spaces), SVD.append(svd)


# # --- For projection ---
# SPACES, SVD = [], []
# max_n = 0
# for block_id, layer_weights in enumerate(proj_layer_weights, start=1):
#     spaces, svd = [], []

#     space_head, svd_head, rank = extract_spaces(layer_weights)
#     svd_head[1], n = thresholding(svd_head[1], 10) # , float('inf'))
#     max_n = max(max_n, n)
#     spaces.append(space_head), svd.append(svd_head)

#     max_sigma = svd_head[1].max()
#     min_sigma = svd_head[1].min()

#     if PRINT:
#         print("#"*36 + f"Transformer Block {block_id:02d}" + "#"*37)
#         print(f"rank:{rank}     new num of σs: {n}  max σ: {max_sigma:.3f}   min σ: {min_sigma:.3f}   (max σ / min σ): {max_sigma/min_sigma:.3f}")

#     SPACES.append(spaces), SVD.append(svd)

# --- Plotting Setup ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 9))

# Define a color for each of the 12 Transformer Blocks
colors = plt.colormaps['tab20'].colors[:12]
# Define a linestyle for each of the 6 Heads
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 10))]

# --- Plotting Loop ---
for block_id, block_svd in enumerate(SVD):
    color = colors[block_id]
    for head_id, head_svd in enumerate(block_svd):
        singular_values = head_svd[1]
        
        # Sort singular values in descending order for a clean plot
        sorted_sv = np.sort(singular_values)[::-1]
        
        # The x-axis is the index of the singular value
        x_axis = np.arange(1, len(sorted_sv) + 1)
        
        # Plot the singular value spectrum for the current head
        ax.plot(
            x_axis,
            sorted_sv,
            color=color,
            linestyle=linestyles[head_id],
            linewidth=2,
            alpha=0.8
        )

# --- Enhancements ---
ax.set_title('Singular Value Spectrum of Attention Heads by Block and Head', fontsize=18)
ax.set_xlabel('Singular Value Index (1 to 64)', fontsize=12)
ax.set_ylabel('Singular Value (σ) - Log Scale', fontsize=12)

# A log scale is ESSENTIAL to see the massive drop-off and compare heads
ax.set_yscale('log')
ax.set_xlim(0, max_n)

# --- Custom Legend ---
# A legend with 72 entries is unreadable. We create two separate legends.
# Legend for Blocks (Colors)
block_legend_elements = [Line2D([0], [0], color=colors[i], lw=4, label=f'Block {i+1}') for i in range(12)]
legend1 = ax.legend(handles=block_legend_elements, loc='upper right', title='Transformer Block')

# Legend for Heads (Linestyles)
head_legend_elements = [Line2D([0], [0], color='black', lw=2, linestyle=ls, label=f'Head {i+1}') for i, ls in enumerate(linestyles)]
ax.legend(handles=head_legend_elements, loc='lower left', title='Attention Head')

# Add the first legend back after creating the second
ax.add_artist(legend1)

plt.tight_layout()
plt.show()


