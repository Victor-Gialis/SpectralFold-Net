# Self-Supervised Learning for Industrial Condition Monitoring

## Overview

This repository investigates the contribution of self-supervised representation learning to vibration-based fault diagnosis under data scarcity constraints.

The work focuses on learning transferable representations from unlabeled vibration spectra using modern self-supervised learning (SSL) paradigms, and on systematically evaluating their impact on downstream tasks relevant to industrial maintenance and condition monitoring.

This project serves as the experimental framework for a PhD thesis in industrial maintenance and applied machine learning.

## Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU support)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/CWRU_MAE.git
cd CWRU_MAE

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as editable package
pip install -e .
```

### Dataset Setup

Download the datasets and place them in `data/raw/`:

- **CWRU**: [Case Western Reserve University Bearing Dataset](https://engineering.case.edu/bearingdatacenter)
- **LASPI**: [Gearbox Fault Dataset](https://www.kaggle.com/)

## Quick Start

### Pretraining with MAE

```bash
python experiments/pretrain_mae.py --config configs/mae_pretrain.yaml
```

### Data Scarcity Evaluation

```bash
python experiments/data_scarcity.py --backbone_init mae --data_ratio 0.1
```

### Downstream Task

```bash
python experiments/downstream_mae.py --config configs/downstream.yaml
```

## Scientific Motivation

In industrial condition monitoring, labeled fault data are often:

scarce,

costly to obtain,

highly imbalanced across operating conditions.

Self-supervised learning offers a promising alternative by exploiting large amounts of unlabeled sensory data. However, the actual benefit of SSL representations in realistic diagnostic scenarios remains insufficiently characterized, especially under strict data scarcity conditions.

This repository addresses the following questions:

How do different self-supervised pretext tasks affect representation quality for mechanical fault diagnosis?

How robust are SSL representations when only a small fraction of labeled data is available?

What is the relative benefit of linear probing versus full fine-tuning?

How sensitive are results to the choice of pretraining dataset and SSL method?

Self-Supervised Methods

The framework supports multiple SSL paradigms, each implemented with method-specific interactions between:

input preprocessing,

backbone encoder,

pretext head,

loss definition.

Currently implemented or planned methods include:

Masked Autoencoders (MAE)

Spectral Aliasing Pretext (SAP)

Momentum Contrast (MoCo) (planned)

Joint Embedding Predictive Architectures (JEPA) (planned)

Each method is treated as a first-class research object rather than a generic wrapper.

Backbone Architectures

The primary backbone architecture is:

ViT-1D: Vision Transformer adapted to 1D vibration spectra

Backbones are:

pre-trained via SSL,

saved independently,

reused for downstream experiments with controlled probing strategies.

Downstream Tasks

Downstream evaluation is designed to be task-agnostic.

Supported tasks include:

Classification (fault type, severity, operating condition)

Regression (fault size, health indicator, degradation proxy)

Downstream heads support:

Linear probing

Non-linear probing

Full fine-tuning (optional)

Each downstream head implements its own loss function, ensuring clean separation between:

representation learning,

task-specific optimization.

Data Scarcity Protocol

A dedicated data_scarcity experiment pipeline enables systematic evaluation under limited labeled data.

Controlled parameters include:

SSL pretraining method

Pretraining dataset

Fine-tuning dataset

Labeled data ratio

Random seed

Probing strategy (linear / non-linear)

Fine-tuning or frozen backbone

This design enables fair and reproducible comparisons across SSL methods.

## Reproducibility

All experiments include:
- Full configuration files (saved automatically)
- Model checkpoints and weights
- Training logs with Weights & Biases integration
- Deterministic seeding for reproducible results

Key random seeds are controlled via:
```bash
python experiments/data_scarcity.py --seed 42 --num_runs 5
```

## Citation

If you use this code in your research, please cite it as:

**BibTeX:**
```bibtex
@software{Grotus2026,
  author = {Grotus, Victor},
  title = {Self-Supervised Learning for Industrial Condition Monitoring},
  year = {2026},
  url = {https://github.com/yourusername/CWRU_MAE},
  note = {GitHub repository}
}
```

**APA:**
```
Grotus, V. (2026). Self-Supervised Learning for Industrial Condition Monitoring. 
Retrieved from https://github.com/yourusername/CWRU_MAE
```

For journal citations with DOI, see [CITATION.cff](CITATION.cff).

## Status

This repository is under active development and continuously extended as part of ongoing doctoral research.

## Author

**Victor Grotus**  
PhD candidate in Industrial Maintenance  
Self-supervised learning for condition monitoring

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Case Western Reserve University for the bearing dataset
- Jean Zay (IDRIS) for HPC supportous train/validation/test splits depending on the experimental stage.

Project Structure
.
├── models/
│   ├── backbone/        # Encoder architectures (e.g. ViT-1D)
│   ├── ssl/             # Self-supervised methods and pretext heads
│   └── downstream/      # Downstream task heads (classification, regression)
│
├── dataset/             # Dataset loaders and preprocessing
├── training/            # Generic training loops (pretrain / downstream)
├── experiments/         # Experiment scripts (pretraining, data scarcity)
├── results/             # Checkpoints, logs, and configurations
└── utils/               # Utilities and shared components

Reproducibility

Each experiment automatically stores:

model checkpoints,

full configuration files (config.json),

training logs and curves.

This allows:

exact reconstruction of pretrained backbones,

systematic reuse in downstream experiments,

transparent comparison across methods.

Execution on HPC (Jean Zay)

The experimental design is compatible with large-scale HPC execution.
Parameter sweeps (SSL method, data ratio, seed, probing strategy) are intended to be launched via job submission scripts using:

from idr_pytools import gpu_jobs_submitter


The Python experiment scripts are kept stateless, ensuring clean integration with SLURM-based workflows.

Status

This repository is under active development and continuously extended as part of ongoing doctoral research.

Author

Victor
PhD candidate in Industrial Maintenance
Self-supervised learning for condition monitoring