# Experiment Result Reproduction of RLoc Paper

> RLoc paper: [RLoc: Towards Robust Indoor Localization by Quantifying Uncertainty](https://dl.acm.org/doi/abs/10.1145/3631437)

This repository reproduces the key experimental results of the RLoc paper, including model training, uncertainty modeling, and evaluation under leave-one-user-out (LOUO) settings.

---

## 1. Overview

RLoc proposes a robust indoor localization framework by explicitly modeling prediction uncertainty.  
Unlike conventional localization models that only output point estimates, RLoc predicts both:

- mean location (μ)
- uncertainty (σ)

and uses probabilistic learning to improve robustness under domain shift (e.g., cross-user scenarios).

This repository focuses on:

- reproducing the RLoc neural model
- implementing leave-one-user-out evaluation
- analyzing prediction uncertainty and localization error
- comparing with baseline methods (e.g., 2D-FFT, triangulation)

---

## 2. File Structure

```{bash}
RLoc/
│
├── dataset/
│ └── human_held_device_wifi_indoor_localization/
│ ├── Conference/
│ ├── Laboratory/
│ ├── Lounge/
│ └── Office/
│
├── reproduction/
│ ├── dataset.py # dataset loading & preprocessing
│ ├── model.py # RLoc neural network architecture
│ ├── train.py # training & evaluation pipeline
│ ├── results/ # experiment outputs (ignored by git)
│ └── pycache/
│
├── demo.m # MATLAB demo for signal processing
├── obtain_parameters.m # parameter extraction
├── orientation_xy.m # coordinate transformation
├── triangulation_min.m # triangulation baseline
│
├── README.md
└── .gitignore
```

### Directory Explanation

#### (1) dataset/

Raw dataset used in the RLoc paper.  
Scenarios correspond to different indoor environments:

- Conference: conference room
- Laboratory: laboratory environment
- Lounge: public lounge
- Office: office space

Each folder contains WiFi sensing data collected from human-held devices.

---

#### (2) reproduction/

Core reproduction codebase.

- `dataset.py`  
  Data loading, normalization, train/test split, and LOUO protocol.
- `model.py`  
  RLoc neural network architecture with dual heads for μ and σ.
- `train.py`  
  Training loop, probabilistic loss function, and evaluation metrics.
- `results/`  
  Saved checkpoints (`.pt`) and logs.  
  ⚠️ Excluded from GitHub due to file size limits.

---

## 3. Experimental Protocol

### 3.1 Leave-One-User-Out (LOUO)

We follow the RLoc paper's cross-user evaluation setting:

- Train on all users except one
- Test on the held-out user
- Repeat for each user

Formally:

$$
\mathcal{D}_{train} = \bigcup_{u \neq u^*} \mathcal{D}_u,\quad
\mathcal{D}_{test} = \mathcal{D}_{u^*}
$$

---

### 3.2 Probabilistic Output Modeling

RLoc predicts a probabilistic location distribution:

$$
\hat{\mathbf{y}} \sim \mathcal{N}(\mu, \sigma^2)
$$

where:

- \(\mu \in \mathbb{R}^2\): predicted position
- \(\sigma \in \mathbb{R}^2\): uncertainty estimation

Loss function (negative log-likelihood):

$$
\mathcal{L} = \sum_i \left( \frac{\|y_i - \mu_i\|^2}{2\sigma_i^2} + \log \sigma_i \right)
$$

---

## 4. Running the Experiments

### 4.1 Environment Setup

```bash
conda create -n rloc python=3.10
conda activate rloc
pip install -r requirements.txt
```

