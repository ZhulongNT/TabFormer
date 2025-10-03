# TabFormer - Pip Installation Guide (No Conda Required)

This guide shows you how to train a Tabular BERT model using **pip** instead of conda, with the **latest Python version (3.12.3)**.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a Tabular BERT Model

**For Credit Card Transaction Data:**
```bash
python main.py --do_train --mlm --field_ce --lm_type bert \
               --field_hs 64 --data_type card \
               --output_dir ./output
```

**For PRSA Air Quality Data:**
```bash
python main.py --do_train --mlm --field_ce --lm_type bert \
               --field_hs 64 --data_type prsa \
               --output_dir ./output
```

## 📋 Complete Package List (pip freeze)

See **`pip_freeze_output.txt`** for the complete list of all installed packages with exact versions.

Key packages:
```
torch==2.8.0+cpu
transformers==4.56.2
pandas==2.3.3
scikit-learn==1.7.2
numpy==2.1.2
tokenizers==0.22.1
huggingface-hub==0.35.3
```

## 🔧 What Was Fixed

The original code used deprecated imports from transformers v3.2.0. All imports have been updated for the latest version (v4.56.2):

### Changes in `models/tabformer_bert.py`:
```python
# OLD (broken with new transformers):
from transformers.modeling_bert import ACT2FN, BertLayerNorm
from transformers.configuration_bert import BertConfig

# NEW (working):
from transformers.models.bert.modeling_bert import ACT2FN
from transformers import BertConfig
# BertLayerNorm → nn.LayerNorm
```

### Changes in `models/tabformer_gpt2.py`:
```python
# OLD (broken):
from transformers.modeling_gpt2 import GPT2LMHeadModel

# NEW (working):
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
```

## ✅ Verification

All components have been tested:
- ✅ Module imports working
- ✅ Model initialization working
- ✅ Forward pass (inference) working
- ✅ Loss computation working
- ✅ Training can start successfully

## 📦 Package Versions Tested

| Package | Version |
|---------|---------|
| Python | 3.12.3 |
| PyTorch | 2.8.0 |
| Transformers | 4.56.2 |
| Pandas | 2.3.3 |
| Scikit-learn | 1.7.2 |
| NumPy | 2.1.2 |

## 📁 New Files Added

- **`requirements.txt`** - Main dependencies with version constraints
- **`requirements-minimal.txt`** - Minimal essential dependencies
- **`pip_freeze_output.txt`** - **Complete pip freeze package list** ⭐
- **`INSTALLATION.md`** - Detailed installation guide
- **`SETUP_SUMMARY.md`** - Quick reference summary
- **`.gitignore`** - Excludes Python cache files

## 🎮 GPU Support

For GPU acceleration, install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then run training with the same command - PyTorch will automatically use GPU if available.

## 🐛 Troubleshooting

### Import Errors
```bash
pip install --upgrade transformers
```

### Memory Issues
Reduce batch size or use smaller field hidden size:
```bash
python main.py --do_train --mlm --field_ce --lm_type bert \
               --field_hs 32 --data_type card \
               --output_dir ./output
```

### Clear Python Cache
```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
```

## 📖 Documentation

- **`INSTALLATION.md`** - Complete installation guide with examples
- **`SETUP_SUMMARY.md`** - Summary of changes and setup
- **`pip_freeze_output.txt`** - Full package list

## 🔗 Original Paper

```bibtex
@inproceedings{padhi2021tabular,
  title={Tabular transformers for modeling multivariate time series},
  author={Padhi, Inkit and Schiff, Yair and Melnyk, Igor and Rigotti, Mattia and Mroueh, Youssef and Dognin, Pierre and Ross, Jerret and Nair, Ravi and Altman, Erik},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={3565--3569},
  year={2021},
  organization={IEEE}
}
```

## ❓ Questions?

For detailed troubleshooting and additional information, see:
- `INSTALLATION.md` - Full installation guide
- `SETUP_SUMMARY.md` - Technical summary of changes
- Original `README.md` - Paper details and dataset information

---

**Updated for:** Python 3.12.3, PyTorch 2.8.0, Transformers 4.56.2  
**No conda required!** 🎉
