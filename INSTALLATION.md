# Installation Guide (Without Conda)

This guide provides instructions for setting up TabFormer using pip instead of conda, with the latest Python version.

## Requirements

- Python 3.12 or higher (tested on Python 3.12.3)
- pip package manager

## Installation Steps

### 1. Install Dependencies

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

Or install them individually:

```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For GPU support, use:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install transformers pandas scikit-learn
```

### 2. Verify Installation

You can verify that everything is installed correctly by running:

```python
python -c "import torch; import transformers; import pandas; import sklearn; print('All imports successful!')"
```

## Training a Tabular BERT Model

### For Credit Card Dataset

```bash
python main.py --do_train --mlm --field_ce --lm_type bert \
               --field_hs 64 --data_type card \
               --output_dir ./output
```

### For PRSA Dataset

```bash
python main.py --do_train --mlm --field_ce --lm_type bert \
               --field_hs 64 --data_type prsa \
               --output_dir ./output
```

## Key Changes from Original Setup

### Fixed Import Issues

The original code was written for older versions of the transformers library (v3.2.0). The following imports have been updated for compatibility with the latest transformers version (v4.56.2):

**In `models/tabformer_bert.py`:**
- Changed: `from transformers.modeling_bert import ACT2FN, BertLayerNorm`
- To: `from transformers.models.bert.modeling_bert import ACT2FN`
- Changed: `from transformers.configuration_bert import BertConfig`
- To: `from transformers import BertConfig`
- Changed: `BertLayerNorm` to `nn.LayerNorm`

**In `models/tabformer_gpt2.py`:**
- Changed: `from transformers.modeling_gpt2 import GPT2LMHeadModel`
- To: `from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel`

## Package Versions

The code has been tested with the following package versions:

- Python: 3.12.3
- PyTorch: 2.8.0
- Transformers: 4.56.2
- Pandas: 2.3.3
- Scikit-learn: 1.7.2
- NumPy: 2.1.2

See `pip_freeze_output.txt` for a complete list of all installed packages and their versions.

## Troubleshooting

### Import Errors

If you encounter import errors related to `transformers`, make sure you have the latest version installed:

```bash
pip install --upgrade transformers
```

### CUDA/GPU Issues

If you want to use GPU acceleration and encounter CUDA-related errors, ensure you have the correct PyTorch version for your CUDA version. Visit [PyTorch's website](https://pytorch.org/get-started/locally/) for the appropriate installation command.

### Memory Issues

If you run out of memory during training, try reducing the batch size or using a smaller model by adjusting `--field_hs` parameter.
