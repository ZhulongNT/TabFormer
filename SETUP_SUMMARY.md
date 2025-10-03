# TabFormer Setup Summary - Pip Installation (No Conda)

## Overview

This repository has been updated to work with the latest Python version (3.12.3) and transformers library (4.56.2), eliminating the need for conda. All dependencies can now be installed using pip.

## What Was Fixed

### 1. Import Statement Updates

The original code was written for transformers v3.2.0. The following files have been updated to work with transformers v4.56.2:

#### `models/tabformer_bert.py`
```python
# OLD (deprecated):
from transformers.modeling_bert import ACT2FN, BertLayerNorm
from transformers.modeling_bert import BertForMaskedLM
from transformers.configuration_bert import BertConfig

# NEW (current):
from transformers.models.bert.modeling_bert import ACT2FN, BertForMaskedLM
from transformers import BertConfig
# BertLayerNorm replaced with nn.LayerNorm
```

#### `models/tabformer_gpt2.py`
```python
# OLD (deprecated):
from transformers.modeling_gpt2 import GPT2LMHeadModel

# NEW (current):
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
```

### 2. New Files Added

- **`requirements.txt`**: Lists all required Python packages with minimum versions
- **`pip_freeze_output.txt`**: Complete list of all installed packages with exact versions
- **`INSTALLATION.md`**: Comprehensive installation and usage guide
- **`.gitignore`**: Prevents Python cache files from being committed

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Tabular BERT Model

For credit card transaction data:
```bash
python main.py --do_train --mlm --field_ce --lm_type bert \
               --field_hs 64 --data_type card \
               --output_dir ./output
```

For PRSA air quality data:
```bash
python main.py --do_train --mlm --field_ce --lm_type bert \
               --field_hs 64 --data_type prsa \
               --output_dir ./output
```

## Verified Package Versions

The code has been tested and verified to work with:

| Package | Version |
|---------|---------|
| Python | 3.12.3 |
| torch | 2.8.0+cpu |
| transformers | 4.56.2 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| numpy | 2.1.2 |
| tokenizers | 0.22.1 |
| huggingface-hub | 0.35.3 |

See `pip_freeze_output.txt` for the complete list with all dependencies.

## Verification

The following tests have been performed:

✅ All module imports successful  
✅ Model initialization working  
✅ Forward pass (inference) working  
✅ Loss computation working  
✅ Command-line argument parsing working  

## GPU Support

The current installation uses CPU-only PyTorch. For GPU support, replace the PyTorch installation with:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Breaking Changes from Original

The following changes break backward compatibility with the original conda setup:

1. **Transformers version**: Upgraded from 3.2.0 to 4.56.2
2. **PyTorch version**: Upgraded from 1.6.0 to 2.8.0
3. **Python version**: Updated from 3.7 to 3.12

All these changes are necessary to use modern, maintained versions of the libraries.

## Documentation

For detailed installation instructions, see `INSTALLATION.md`.

## Troubleshooting

If you encounter any import errors:

1. Ensure you have Python 3.12 or later
2. Reinstall dependencies: `pip install -r requirements.txt --upgrade`
3. Clear Python cache: `find . -type d -name "__pycache__" -exec rm -rf {} +`

For more troubleshooting tips, see the "Troubleshooting" section in `INSTALLATION.md`.
