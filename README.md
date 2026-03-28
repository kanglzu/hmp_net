# HMP-Net: Hierarchical Multi-Prior Network for Brain Tumor Segmentation

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.3](https://img.shields.io/badge/pytorch-2.3-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **HMP-Net: A Hierarchical Multi-Prior Network for Brain Tumor Segmentation Integrating Physics, Topology, and Tumor Dynamics**.

<p align="center">
  <img src="/plots/pipeline.pdf" width="100%"/>
</p>

## Highlights

- **Physics-aware fusion**: a learnable coupling matrix models inter-modal MRI dependencies derived from shared tissue parameters.
- **Topology-guided encoding**: differentiable Betti-number approximation via multi-scale morphological gradients captures tumor connectivity and boundary complexity.
- **Dynamics-informed decoding**: a single-step Fisher-Kolmogorov solver embeds biologically plausible reaction-diffusion growth patterns.
- **Hierarchical alignment**: each prior is injected at the semantic level where it is most effective (shallow/mid/deep), retaining 99.6% of exhaustive-deployment accuracy at 53% of the parameters.

## Architecture

<p align="center">
  <img src="/plots/overall.pdf" width="100%"/>
</p>
<p align="center">
  <img src="/plots/module.pdf" width="100%"/>
</p>

| Module | Level | Role |
|--------|-------|------|
| PSE / PGD | Encoder L1 / Decoder L2 | Physics-constrained cross-modal fusion |
| TSA / TGD | Encoder L2 / Decoder L3 | Topology-aware structural encoding |
| TDM / DGD | Encoder L4 / Decoder L4 | Reaction-diffusion dynamics modeling |
| CMF | Before encoder | Cross-Modal Fusion with learnable modality weights |
| EMP | Skip connections | Enhanced Multi-Prior cross-attention gating |

## Installation

```bash
git clone https://github.com/kanglzu/hmp_net.git
cd hmp_net

# create environment
conda create -n hmpnet python=3.12 -y
conda activate hmpnet

# install dependencies
pip install torch==2.3.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install nibabel SimpleITK scipy numpy torchio
```

## Data Preparation

Download BraTS 2021 and/or BraTS 2018 from [Synapse](https://www.synapse.org/#!Synapse:syn25829067) and organize as:

```
data/
├── BraTS2021/
│   ├── BraTS2021_00000/
│   │   ├── BraTS2021_00000_t1.nii.gz
│   │   ├── BraTS2021_00000_t1ce.nii.gz
│   │   ├── BraTS2021_00000_t2.nii.gz
│   │   ├── BraTS2021_00000_flair.nii.gz
│   │   └── BraTS2021_00000_seg.nii.gz
│   └── ...
└── BraTS2018/
    └── ...
```

**Label mapping:**
- 0: Background
- 1: Necrotic / Non-enhancing tumor core (NCR/NET)
- 2: Peritumoral edema (ED)
- 4: Enhancing tumor (ET)

**Evaluation regions:** WT = {1, 2, 4}, TC = {1, 4}, ET = {4}

## Project Structure

```
.
├── codes/
│   ├── models/
│   │   ├── hmpnet.py          # Main network (encoder-decoder + CMF)
│   │   ├── pse.py             # Physical Signal Encoder
│   │   ├── tsa.py             # Topological Structure Analyzer
│   │   ├── tdm.py             # Tumor Dynamics Modeler
│   │   ├── pgd.py             # Physics-Guided Decoder
│   │   ├── tgd.py             # Topology-Guided Decoder
│   │   ├── dgd.py             # Dynamics-Guided Decoder
│   │   └── emp_skip.py        # Enhanced Multi-Prior Skip Connection
│   └── losses/
│       ├── combined_loss.py   # Dice + Focal + Deep Supervision
│       └── prior_losses.py    # Physics / Topology / Dynamics regularizers
├── data/
│   └── brats_dataset.py       # BraTS 2018 & 2021 data loader
└── hmpnetpaper/               # LaTeX source and figures
```

## Training

```bash
python train.py \
    --data_dir data/BraTS2021 \
    --epochs 400 \
    --batch_size 2 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --lambda_focal 1.0 \
    --lambda_phys 0.1 \
    --lambda_topo 0.1 \
    --lambda_dyn 0.1
```

**Key hyperparameters:**

| Category | Parameter | Value |
|----------|-----------|-------|
| Optimizer | AdamW + cosine annealing | lr=1e-4, wd=1e-4 |
| Loss weights | λ_focal / λ_phys / λ_topo / λ_dyn | 1.0 / 0.1 / 0.1 / 0.1 |
| Regularization | τ_phys / τ_topo / μ_dyn | 0.3 / 0.05 / 0.5 |
| Architecture | Ghost ratio / SE ratio / Attention ratio | 2 / 16 / 8 |
| Input | Crop size / Modalities | 128³ / 4 (T1, T1ce, T2, FLAIR) |

## Inference

```bash
python test.py \
    --data_dir data/BraTS2021 \
    --checkpoint checkpoints/hmpnet_best.pth
```

