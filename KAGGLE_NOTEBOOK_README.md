# TabFormer Kaggle Notebook

## Overview

`tabformer_kaggle_notebook.ipynb` is a comprehensive, single-file Jupyter notebook that runs TabFormer (Tabular Transformer) in the Kaggle environment. This notebook contains all necessary code merged linearly for easy execution.

## Features

- ✅ **Single File**: All code in one notebook - no external dependencies
- ✅ **Kaggle Optimized**: Uses Kaggle-specific paths (`/kaggle/input/`, `/kaggle/working/`)
- ✅ **CPU Compatible**: Configured to run on CPU (no GPU required)
- ✅ **Auto-Install**: Automatically installs `transformers[torch]` and dependencies
- ✅ **Well-Commented**: Includes explanatory comments throughout
- ✅ **IPython Display**: Uses IPython display for better visualization
- ✅ **Complete Pipeline**: From data loading to model training

## Quick Start on Kaggle

### 1. Upload Data

Upload your credit card transaction CSV file to Kaggle as a dataset. The CSV should have these columns:

```
User, Card, Year, Month, Day, Time, Amount, Use Chip, Merchant Name, 
Merchant City, Merchant State, Zip, MCC, Errors?, Is Fraud?
```

### 2. Upload Notebook

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Click "File" → "Upload Notebook"
4. Upload `tabformer_kaggle_notebook.ipynb`

### 3. Add Input Data

1. In the notebook, click "Add Data" on the right sidebar
2. Select your uploaded dataset
3. The data will be available in `/kaggle/input/your-dataset-name/`

### 4. Configure Paths

In **Cell 13 (Configuration)**, update the paths:

```python
class Config:
    # Update this to match your dataset path
    data_root = "/kaggle/input/your-dataset-name/"
    data_fname = "your_file_name"  # without .csv extension
    
    # These are fine as defaults
    output_dir = "/kaggle/working/checkpoints"
    log_dir = "/kaggle/working/logs"
    
    # Optional: Limit rows for faster testing
    nrows = 10000  # Set to None to use all data
    
    # Training parameters
    num_train_epochs = 3
    field_hs = 64
```

### 5. Run All Cells

1. Click "Run All" or run cells sequentially
2. The notebook will:
   - Install required packages
   - Load and preprocess data
   - Train the TabFormer model
   - Save outputs to `/kaggle/working/`

## Notebook Structure

The notebook contains **33 cells** organized as follows:

| Section | Description |
|---------|-------------|
| **1-2** | Package installation and imports |
| **3-12** | Core TabFormer components (utilities, vocab, models, dataset) |
| **13** | Configuration (⚠️ **Update this with your data paths**) |
| **14** | Main training function |
| **15** | Execute training |
| **16** | View results and outputs |

## Configuration Options

### Model Settings

```python
# Model architecture
lm_type = "bert"          # Use "bert" or "gpt2"
mlm = True                # Masked language modeling (required for BERT)
field_ce = True           # Field-wise cross-entropy loss
flatten = False           # Use hierarchical structure (False) or flat (True)
field_hs = 64             # Field hidden size (64 for faster, 768 for better quality)

# Training
num_train_epochs = 3      # Number of training epochs
save_steps = 500          # Save checkpoint every N steps
stride = 5                # Sliding window stride
```

### Data Settings

```python
nrows = 10000             # Limit rows (None = all data)
cached = False            # Use cached preprocessed data
user_ids = None           # Filter specific users (None = all users)
skip_user = False         # Exclude user field from model
```

## Expected Outputs

After training completes, you'll find:

### `/kaggle/working/checkpoints/`
- Model checkpoints (every `save_steps` steps)
- `final_model/` - Final trained model
- `vocab.nb` - Vocabulary file

### `/kaggle/working/logs/`
- Training logs

## Memory and Performance Tips

### For Limited Memory:
```python
nrows = 5000              # Use fewer rows
field_hs = 32             # Reduce hidden size
num_train_epochs = 1      # Fewer epochs
```

### For Better Quality:
```python
nrows = None              # Use all data
field_hs = 128            # Larger hidden size
num_train_epochs = 10     # More epochs
```

## Model Types

### BERT (Recommended)
```python
lm_type = "bert"
mlm = True
field_ce = True
```
- Bidirectional context
- Masked language modeling
- Better for understanding patterns

### GPT-2
```python
lm_type = "gpt2"
mlm = False
field_ce = True
```
- Autoregressive generation
- Left-to-right context
- Good for sequence generation

## Troubleshooting

### Issue: "File not found"
**Solution**: Check `data_root` and `data_fname` in Configuration (Cell 13)

### Issue: "Out of memory"
**Solution**: Reduce `nrows`, `field_hs`, or `num_train_epochs`

### Issue: "CUDA error" 
**Solution**: The notebook is configured for CPU. If you see this, ensure Cell 1 runs properly.

### Issue: "Column not found"
**Solution**: Ensure your CSV has the correct column names (see Quick Start #1)

### Issue: "Import error"
**Solution**: Ensure Cell 3 (package installation) completed successfully. May need to restart kernel.

## Data Format

Your input CSV must contain these columns in any order:

| Column | Type | Description |
|--------|------|-------------|
| User | int | User ID |
| Card | int | Card ID |
| Year | int | Transaction year |
| Month | int | Transaction month (1-12) |
| Day | int | Transaction day |
| Time | str | Time in "HH:MM" format |
| Amount | str | Amount with $ sign (e.g., "$50.00") |
| Use Chip | str | "Chip Transaction" or "Swipe Transaction" |
| Merchant Name | str | Merchant name |
| Merchant City | str | City name |
| Merchant State | str | State code |
| Zip | int/float | ZIP code (can have NaN) |
| MCC | int | Merchant Category Code |
| Errors? | str | Error description or NaN |
| Is Fraud? | str | "Yes" or "No" |

### Example Row:
```csv
User,Card,Year,Month,Day,Time,Amount,Use Chip,Merchant Name,Merchant City,Merchant State,Zip,MCC,Errors?,Is Fraud?
0,0,2002,9,1,06:21,$134.09,Swipe Transaction,Merchant 1,City A,NY,12345,3000,,No
```

## CPU vs GPU

The notebook is configured to run on **CPU only** for maximum compatibility:
- Works on any Kaggle notebook (free tier)
- No GPU quota required
- Slower but accessible to everyone

Training time on CPU (approximate):
- 10,000 rows: ~10-15 minutes
- 50,000 rows: ~30-45 minutes
- 100,000 rows: ~1-2 hours

## Next Steps

After training:

1. **Download Model**: Download `/kaggle/working/checkpoints/final_model/` for later use
2. **Evaluate**: Use the saved model for downstream tasks (fraud detection, etc.)
3. **Tune**: Adjust hyperparameters and retrain
4. **Share**: Save and share your notebook with the community

## References

- **Paper**: [TabFormer: Tabular Data Modeling via Transformers](https://arxiv.org/abs/2010.11653)
- **Original Repo**: https://github.com/ZhulongNT/TabFormer

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review notebook comments
3. Check original repository documentation
4. Open an issue on GitHub

## License

This notebook is part of the TabFormer project. See the repository LICENSE file for details.

---

**Happy Training! 🚀**
