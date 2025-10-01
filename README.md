# Transformer Analysis for DINOv3 🤖

This repository contains a suite of tools for visualizing and analyzing the internal workings of the DINOv3 Vision Transformer. It was developed for the SSIAI 2026 project to provide deeper insights into how these models interpret and process image data.

Here you can find scripts to:
* **Visualize Self-Attention**: Interactively select a point on an image and see what other regions the model pays attention to.
* **Visualize Feature Maps**: See how the model semantically segments an image using PCA on its deep features.
* **Extract Model Weights**: Inspect the core parameters from the Query-Key-Value layers.

---

## 🎯 Getting Started

Follow these steps to set up your environment and run the analysis scripts.

### **1. Prerequisites**

* A `conda` environment manager.
* `git` installed on your system.

### **2. Installation**

First, clone this repository and create the conda environment.

```shell
# Clone this project
git clone https://github.com/SebastianJanampa/TransformerAnalysis.git
cd TransformerAnalysis

# Create and activate the conda environment
conda create -n ssiai python=3.12
conda activate ssiai
```

Next, clone the official DINOv3 repository and install all necessary requirements.
```shel
# Clone the DINOv3 dependency
git clone https://github.com/facebookresearch/dinov3.git

# Install Python packages
pip install -r requirements.txt

# Install DINOv3 as an editable package
cd dinov3
pip install -e .
cd ..
```

### **3. Download Model Weights**

You must download the pre-trained model checkpoints directly from the official **[DINOv3 repository](https://github.com/facebookresearch/dinov3?tab=readme-ov-file#pretrained-models)**.

Create a `weights` directory and place the `.pth` files inside, following this structure:
```
TransformerAnalysis/
├── weights/
│   └── dinov3/
│       └── dinov3_vits16_pretrain.pth
├── dinov3/
├── images/
└── utils/
```

---

## 🚀 Usage Examples

Here’s how to run the main visualization scripts.

### **Visualizing Self-Attention**

This script lets you interactively click on a point in an image to generate its corresponding self-attention map. This reveals which parts of the image the model considers most relevant to the point you selected.

```shell
python self_attention_viz.py
```
### **Visualizing Feature Maps**

This script visualizes the model's deep features using two methods:
1.  **PCA Visualization**: Compresses the features into a 3-channel image, revealing the model's unsupervised semantic segmentation.
2.  **Single Feature Map**: Shows a heatmap for a single feature's activation.

```shell
python feature_map_viz.py
```

### **Extracting QKV Layer Weights**

This utility script inspects the model's internal architecture by extracting the learned **Query-Key-Value (QKV)** weights and biases from each transformer block. It then prints their shapes to the console.

```shell
python qkv_layer_extraction.py
```


