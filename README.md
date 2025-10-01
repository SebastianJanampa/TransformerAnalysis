<h2 align="center">
  Tranformer Analysis: This repo is the code for SSIAI 2026
</h2>


## 🚀 Updates
- [x] **\[2025.09.30\]** Upload code for qkv layer visualization.


### Setup

```shell
# Create environment
conda create -n ssiai python=3.12
conda activate ssiai
```

### Requirements
```shell
# Download DINOv3 repo
git clone https://github.com/facebookresearch/dinov3.git
# Install requirements
pip install -r requirements.txt
cd dinov3
pip install -e .
cd ..
```
### DINOv3 weights
Please, refer to [DINOv3](https://github.com/facebookresearch/dinov3?tab=readme-ov-file#pretrained-models) to download the model checkpoints. 
Store the `.pt` file on this way
```
── dinov3
├── checkpoints
|  └── dinov3
|  |   └── <checkpoints.pth>
├── images
└── utils
```



