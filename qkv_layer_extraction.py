import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import StrMethodFormatter, PercentFormatter
import math

def plot_histogram_on_axes(vector, axes, plot_label, color, darker_color, shared_x_range, shared_y_range):
    num_bins = 75

    plot_range = shared_x_range if shared_x_range else (vector.min(), vector.max())
    bin_width = (plot_range[1] - plot_range[0]) / num_bins

    weights = np.ones_like(vector) / len(vector)

    axes.hist(vector, bins=num_bins, weights=weights, color=color, alpha=0.5,
              edgecolor='white', label=plot_label, zorder=2, range=plot_range)

    kde = gaussian_kde(vector)
    x_range = np.linspace(plot_range[0], plot_range[1], 200)
    kde_scaled = kde(x_range) * bin_width
    axes.plot(x_range, kde_scaled, color=darker_color, linewidth=2, zorder=3)

    mean_val = np.mean(vector)
    std_val = np.std(vector)
    return mean_val, std_val
    
def plot_weights_with_histograms_horizontal(weight_tensor_subset, num_heads_in_figure, block_idx, start_head_idx, shared_vmin, shared_vmax, shared_x_range, shared_y_range):
    """
    Plots a single figure for a subset of attention heads with a dedicated column for row labels.
    """
    Wq_all, Wk_all, Wv_all = weight_tensor_subset.unbind(dim=1)

    # --- CHANGED: Add an extra column for row titles ---
    num_plot_cols = num_heads_in_figure + 1
    # Make the first column (for titles) much narrower than the plot columns
    width_ratios = [0.2] + [1] * num_heads_in_figure

    # 1. Set up the figure with the new grid dimensions
    fig, axes = plt.subplots(
        5, num_plot_cols,
        figsize=(5 * num_heads_in_figure + 1, 10.5), # Add a little extra width for the new column
        gridspec_kw={
            'height_ratios': [1, 1, 1, 0.8, 0.3],
            'width_ratios': width_ratios
        },
        constrained_layout=True
    )
    if num_heads_in_figure == 1:
        axes = np.expand_dims(axes, axis=1)

    end_head_idx = start_head_idx + num_heads_in_figure
    fig.suptitle(f"Analysis of Block {block_idx:02d} (Heads {start_head_idx+1}-{end_head_idx})", fontsize=16)

    # 2. Plot the heatmaps, offsetting by one column
    row_titles = ["Wq Weights", "Wk Weights", "Wv Weights"]
    all_weights_types = [Wq_all, Wk_all, Wv_all]

    for row_idx, W_type in enumerate(all_weights_types):
        # --- ADDED: Place row title in the dedicated first column ---
        title_ax = axes[row_idx, 0]
        title_ax.axis('off')
        title_ax.text(0.5, 0.5, row_titles[row_idx], ha='center', va='center', rotation=90, fontsize=12)

        # Plot heatmaps in the subsequent columns
        for local_head_idx in range(num_heads_in_figure):
            global_head_idx = start_head_idx + local_head_idx
            # --- CHANGED: Plot into column local_head_idx + 1 ---
            ax = axes[row_idx, local_head_idx + 1]
            im = ax.imshow(W_type[local_head_idx].cpu(), cmap='viridis', vmin=shared_vmin, vmax=shared_vmax)
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(f"Head {global_head_idx+1}", fontsize=14, pad=10)

    # 3. Create the colorbar (this logic remains the same)
    rightmost_heatmap_axes = axes[0:3, -1].tolist()
    fig.colorbar(im, ax=rightmost_heatmap_axes, label="Weight Value", location='right', aspect=40, pad=0.04)

    # 4. Plot the histograms, offsetting by one column
    # Make the title axes in the histogram row invisible
    axes[3, 0].axis('off')
    for local_head_idx in range(num_heads_in_figure):
        global_head_idx = start_head_idx + local_head_idx
        # --- CHANGED: Plot into column local_head_idx + 1 ---
        ax_hist = axes[3, local_head_idx + 1]
        ax_hist.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
        ax_hist.spines[['top', 'right']].set_visible(False)

        if local_head_idx == 0:
            ax_hist.set_ylabel("Percentage (%)", fontsize=9)
        ax_hist.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax_hist.xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))

        ax_hist.set_xlim(shared_x_range)
        ax_hist.set_ylim(shared_y_range)
        y_ticks = np.linspace(shared_y_range[0], shared_y_range[1], 6)
        ax_hist.set_yticks(y_ticks)

        wq_data = Wq_all[local_head_idx].flatten().cpu().numpy()
        wk_data = Wk_all[local_head_idx].flatten().cpu().numpy()
        wv_data = Wv_all[local_head_idx].flatten().cpu().numpy()

        wq_mean, wq_std = plot_histogram_on_axes(wq_data, ax_hist, 'Wq', "#4c72b0", "#2a4b7c", shared_x_range, shared_y_range)
        wk_mean, wk_std = plot_histogram_on_axes(wk_data, ax_hist, 'Wk', "#ff7f0e", "#b25a0a", shared_x_range, shared_y_range)
        wv_mean, wv_std = plot_histogram_on_axes(wv_data, ax_hist, 'Wv', "#2ca02c", "#1f7a1f", shared_x_range, shared_y_range)

        stats_text = (f"Wq: M={wq_mean:.3f}, S={wq_std:.3f}\n"
                      f"Wk: M={wk_mean:.3f}, S={wk_std:.3f}\n"
                      f"Wv: M={wv_mean:.3f}, S={wv_std:.3f}")
        ax_hist.text(0.98, 0.98, stats_text, transform=ax_hist.transAxes, fontsize=10,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    # 5. Create the shared legend
    handles, labels = ax_hist.get_legend_handles_labels()
    for ax in axes[4, :]:
        ax.axis('off')
    # --- CHANGED: Center the legend in the new grid width ---
    legend_ax = axes[4, num_plot_cols // 2]
    legend_ax.legend(handles, labels, loc='center', ncol=3, fontsize=15, frameon=True, edgecolor='gray')

    plt.show()
    
def generate_plots_for_block(weight_tensor, num_heads_total, block_idx, heads_per_figure=3):
    """
    Orchestrator function. Pre-calculates shared ranges across all heads
    and then calls the plotting function for chunks of 3 heads.
    """
    Wq_all, Wk_all, Wv_all = weight_tensor.unbind(dim=1)

    # 1. Pre-calculate shared ranges across ALL heads in the block
    shared_vmin, shared_vmax = weight_tensor.min().item(), weight_tensor.max().item()
    
    global_xmin = weight_tensor.min().item()
    global_xmax = weight_tensor.max().item()
    shared_x_range = (global_xmin - 0.05 * (global_xmax - global_xmin),
                      global_xmax + 0.05 * (global_xmax - global_xmin))

    num_bins = 75
    max_frac = 0
    for head_idx in range(num_heads_total):
        wq_data = Wq_all[head_idx].flatten().cpu().numpy()
        wk_data = Wk_all[head_idx].flatten().cpu().numpy()
        wv_data = Wv_all[head_idx].flatten().cpu().numpy()
        wq_counts, _ = np.histogram(wq_data, bins=num_bins, range=shared_x_range)
        wk_counts, _ = np.histogram(wk_data, bins=num_bins, range=shared_x_range)
        wv_counts, _ = np.histogram(wv_data, bins=num_bins, range=shared_x_range)
        max_frac = max(max_frac, (wq_counts / len(wq_data)).max(), (wk_counts / len(wk_data)).max(), (wv_counts / len(wv_data)).max())
    
    max_percent = max_frac * 100
    padded_max_percent = max_percent + 0.01
    if padded_max_percent == 0: final_ymax_frac = 0.1
    else:
        desired_ticks = 5
        interval = math.ceil((padded_max_percent / desired_ticks) / 5) * 5
        if interval == 0: interval = 5
        final_ymax_percent = math.ceil(padded_max_percent / interval) * interval
        if final_ymax_percent == 0: final_ymax_percent = 5
        final_ymax_frac = final_ymax_percent / 100
    shared_y_range = (0, final_ymax_frac)

    # 2. Loop through heads in chunks and create one figure per chunk
    for start_idx in range(0, num_heads_total, heads_per_figure):
        end_idx = min(start_idx + heads_per_figure, num_heads_total)
        W_subset = weight_tensor[start_idx:end_idx]
        num_heads_in_this_figure = W_subset.shape[0]

        print(f"--- Generating plot for Block {block_idx:02d}, Heads {start_idx+1}-{end_idx} ---")
        plot_weights_with_histograms_horizontal(
            W_subset,
            num_heads_in_this_figure,
            block_idx,
            start_idx,
            shared_vmin,
            shared_vmax,
            shared_x_range,
            shared_y_range
        )


REPO_DIR = "./dinov3"
CHECKPOINT = "weights/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

device = 'cpu'
model = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=CHECKPOINT)
model = model.to(device)

num_heads = model.num_heads
embed_dim = model.embed_dim

qkv_layer_weights = []
for block in model.blocks:
    weights = block.attn.qkv.weight.detach()
    weights = weights.view(num_heads, 3, embed_dim // num_heads, embed_dim)
    qkv_layer_weights.append(weights)

HEADS_PER_FIGURE = 2
for block_idx, W in enumerate(qkv_layer_weights, start=1):
    print(f"#### Analyzing weights for Block {block_idx:02d}")
    print(f"\tFull Shape (num_heads, 3, head_dim, embed_dim): {W.shape}")
    generate_plots_for_block(W, num_heads, block_idx, HEADS_PER_FIGURE)
