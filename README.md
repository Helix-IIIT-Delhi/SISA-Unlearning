<div align="center">

# SISA++: Machine Unlearning for Paralinguistic Speech Processing

## Towards Machine Unlearning for Paralinguistic Speech Processing

[![Paper](https://img.shields.io/badge/Paper-Interspeech%202025-blue)](https://www.isca-archive.org/interspeech_2025/phukan25g_interspeech.html)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-ee4c2c)](https://pytorch.org/)

</div>

---

## Paper Details

**Title**: Towards Machine Unlearning for Paralinguistic Speech Processing

**Authors**: Orchid Chetia Phukan, Girish, Mohd Mujtaba Akhtar, Shubham Singh, Swarup Ranjan Behera, Vandana Rajan, Muskaan Singh, Arun Balaji Buduru, Rajesh Sharma

**Conference**: Interspeech 2025

---

## Abstract

Machine Unlearning (MU) is an emerging privacy technique that enables removal of specific data influence from trained models. This paper is the first to study Machine Unlearning for Paralinguistic Speech Processing (PSP), focusing on Speech Emotion Recognition (SER) and Depression Detection (DD).

We propose SISA++, an extension of SISA that merges shard models using weight averaging for improved performance preservation. We demonstrate on CREMA-D (SER) and E-DAIC (DD) datasets that SISA++ retains more performance than SISA after unlearning.

---

## Overview

**SISA++** introduces privacy-preserving machine unlearning for speech tasks while maintaining model performance.

---

## Key Features

🔐 **Privacy-Preserving** - Remove specific data influence
⚡ **Efficient** - Avoid full retraining
🎯 **Targeted** - Unlearn specific speakers
📊 **Practical Recipes** - Actionable guidance
🗣️ **Speech-Specific** - Optimized for paralinguistic tasks

---

## Quick Start

### Installation

```bash
git clone https://github.com/Helix-IIIT-Delhi/SISA-Unlearning.git
cd SISA-Unlearning
pip install -r requirements.txt
```

### Basic Usage

```python
from sisa import SISAPlusUnlearner

# Initialize SISA++
unlearner = SISAPlusUnlearner(num_shards=4, model_type='SER')

# Train shard models
unlearner.train_shards(train_data, labels)

# Unlearning
unlearner.unlearn('speaker:S001')

# Evaluate
performance = unlearner.evaluate(test_data, test_labels)
print(f"Accuracy: {performance['accuracy']:.3f}")
```

---

## Datasets

### CREMA-D (Speech Emotion Recognition)
- Speakers: 91
- Emotions: 6 classes
- Clips: 7,442 audio samples

### E-DAIC (Depression Detection)
- Participants: 189
- Task: Binary classification
- Format: Conversational interviews

---

## Results

| Method | SER Accuracy | DD Accuracy | Loss |
|--------|-------------|------------|------|
| Baseline | 72.5% | 68.3% | - |
| SISA | 69.8% | 65.7% | -2.7% |
| **SISA++** | **71.2%** | **67.1%** | **-1.3%** |

---

## Repository Structure

```
SISA-Unlearning/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
│
├── sisa/
│   ├── __init__.py
│   ├── unlearner.py          # SISA++ implementation
│   ├── sharding.py           # Data sharding
│   ├── merger.py             # Weight averaging
│   ├── metrics.py            # Evaluation metrics
│   └── utils.py             # Utilities
│
├── models/
│   ├── ser_model.py          # SER models
│   └── dd_model.py           # Depression detection
│
├── datasets/
│   ├── crema_d.py            # CREMA-D loader
│   └── edaic.py              # E-DAIC loader
│
└── scripts/
    ├── train.py
    ├── unlearn.py
    └── evaluate.py
```

---

## Citation

If you find this work useful, please consider citing us:

```bibtex
@inproceedings{phukan2025sisa,
  title={Towards Machine Unlearning for Paralinguistic Speech Processing},
  author={Phukan, Orchid Chetia and Girish and Akhtar, Mohd Mujtaba and Singh, Shubham and Behera, Swarup Ranjan and Rajan, Vandana and Singh, Muskaan and Buduru, Arun Balaji and Sharma, Rajesh},
  booktitle={Proceedings of Interspeech 2025},
  year={2025}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Advancing Privacy-Preserving Machine Learning for Speech Processing**

</div>