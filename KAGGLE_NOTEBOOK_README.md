# TabFormer Kaggle Notebook Guide

## Overview

`tabformer_kaggle_notebook.ipynb` is a self-contained Jupyter notebook that trains a hierarchical TabFormer BERT model on credit card transaction data. It's designed to run in Kaggle's environment with GPU acceleration.

## Features

✅ **Single File**: All code merged into one notebook - no external dependencies  
✅ **Kaggle Ready**: Installs required packages automatically  
✅ **GPU Accelerated**: Uses CUDA when available with FP16 mixed precision  
✅ **IPython Display**: Rich output with HTML/Markdown displays  
✅ **Well Documented**: Comprehensive comments and explanations  
✅ **Production Ready**: Complete pipeline from data loading to model saving

## What's Included

The notebook includes all necessary components:

1. **Package Installation** - Automatically installs PyTorch, Transformers, and dependencies
2. **Vocabulary System** - Field-aware tokenization for tabular data
3. **Dataset Processing** - Credit card transaction encoding and sequence preparation
4. **Model Architecture** - Complete TabFormer BERT implementation:
   - Hierarchical field embeddings
   - Custom BERT configuration
   - Field-wise cross-entropy support
5. **Data Collator** - Masked language modeling for tabular data
6. **Training Pipeline** - Hugging Face Trainer with GPU optimization
7. **Progress Tracking** - Visual feedback with IPython display

## Quick Start

### Running in Kaggle

1. **Upload to Kaggle**:
   - Go to [Kaggle Notebooks](https://www.kaggle.com/code)
   - Click "New Notebook" → "Upload Notebook"
   - Select `tabformer_kaggle_notebook.ipynb`

2. **Upload Data**:
   - Add the credit card dataset as input data
   - Place `card_transaction.v1.csv` in `/kaggle/input/credit-card/data/credit_card/`
   - Or upload the `transactions.tgz` file and extract it

3. **Configure GPU**:
   - In notebook settings, enable GPU accelerator
   - Recommended: GPU P100 or T4

4. **Run All Cells**:
   - Click "Run All" or run cells sequentially
   - Training will start automatically

### Configuration Options

Edit the `CONFIG` dictionary in cell 3 to customize:

```python
CONFIG = {
    'seed': 42,                    # Random seed for reproducibility
    'field_hidden_size': 64,       # Model hidden size (64 for faster, 768 for better quality)
    'seq_len': 10,                 # Number of transactions per sequence
    'stride': 5,                   # Sliding window stride
    'nrows': 100000,               # Number of data rows (None for all)
    'num_train_epochs': 3,         # Training epochs
    'batch_size': 32,              # Batch size (adjust based on GPU memory)
}
```

## Hardware Requirements

### Minimum
- **GPU**: Any CUDA-compatible GPU
- **RAM**: 8 GB
- **Storage**: 5 GB

### Recommended (for faster training)
- **GPU**: NVIDIA P100, V100, or T4
- **RAM**: 16 GB
- **Storage**: 10 GB

## Expected Runtime

With default configuration (`nrows=100000`, `epochs=3`, `batch_size=32`):

| Hardware | Approximate Time |
|----------|-----------------|
| GPU P100 | ~15-20 minutes |
| GPU T4   | ~20-30 minutes |
| CPU      | ~2-3 hours (not recommended) |

For full dataset (~24M rows): 4-8 hours on GPU

## Output

The notebook produces:

1. **Model Checkpoints**: Saved in `./output/checkpoint-*`
2. **Final Model**: Saved in `./output/final_model/`
3. **Vocabulary**: Saved as `./output/vocab.nb`
4. **Training Logs**: Console output with loss and metrics
5. **Summary Statistics**: Dataset info, model parameters, training config

## Model Architecture

**Hierarchical TabFormer BERT**:
- Field-level embeddings for tabular structure
- Multi-head attention across transactions
- Masked language modeling objective
- Field-wise cross-entropy loss

**Key Differences from Standard BERT**:
- Preserves tabular structure (fields)
- Custom tokenization per column
- Hierarchical attention mechanism

## Data Format

The notebook expects credit card transaction CSV with columns:
- `User`, `Card`, `Year`, `Month`, `Day`, `Time`
- `Amount`, `Use Chip`, `Merchant Name`, `Merchant City`, `Merchant State`
- `Zip`, `MCC`, `Errors?`, `Is Fraud?`

## Troubleshooting

### Out of Memory Error
- Reduce `batch_size` in CONFIG (try 16 or 8)
- Reduce `field_hidden_size` (try 32)
- Reduce `nrows` to train on less data

### Slow Training
- Enable GPU in Kaggle notebook settings
- Increase `batch_size` if GPU memory allows
- Reduce `nrows` for faster iteration

### Data Not Found
- Check data path in CONFIG['data_root']
- Ensure CSV file exists at the correct location
- Verify file name matches CONFIG['data_fname']

### Import Errors
- Restart kernel and run cell 1 (package installation)
- Check internet connectivity for package downloads

## Advanced Usage

### Custom Dataset

To use your own tabular data:

1. Format CSV with transaction sequences
2. Update `TransactionDataset.encode_data()` for your columns
3. Adjust `CONFIG` parameters for your data size
4. Modify preprocessing in data encoding section

### Hyperparameter Tuning

Key parameters to tune:
- `field_hidden_size`: Model capacity (32-768)
- `num_train_epochs`: Training duration (1-10)
- `mlm_prob`: Masking probability (0.10-0.20)
- `batch_size`: Training batch size (8-64)
- `seq_len`: Sequence length (5-20)

### Export for Inference

After training, load the model:

```python
from transformers import BertForMaskedLM

model = BertForMaskedLM.from_pretrained('./output/final_model/')
```

## References

- **Paper**: [Tabular Transformers for Modeling Multivariate Time Series](http://arxiv.org/abs/2011.01843)
- **Original Repo**: [IBM/TabFormer](https://github.com/IBM/TabFormer)
- **Hugging Face**: [Transformers Documentation](https://huggingface.co/docs/transformers)

## License

This notebook follows the same license as the TabFormer repository.

## Questions?

For issues or questions:
1. Check the troubleshooting section above
2. Review the inline comments in the notebook
3. Open an issue in the GitHub repository

---

**Note**: This notebook is self-contained and includes all necessary code. You don't need to install or import anything from the repository - just upload the notebook and data to Kaggle!
