# PySR Runner Processing Modes Guide

## Overview

The PySR Runner supports three distinct processing modes, each designed for different data organization scenarios.

## Processing Modes

### 1. `separate_validation` Mode

**When to use:**
- You have pre-split training and validation datasets in separate files
- You want to maintain consistent train/validation splits across experiments
- Your data has already been preprocessed with specific train/val partitions

**Required configuration:**
```json
{
  "runtime_params": {
    "processing_mode": "separate_validation",
    "train_directory": "/path/to/train/data",
    "val_directory": "/path/to/validation/data",
    "training_datasets": ["train1.csv", "train2.csv"],
    "validation_datasets": ["val1.csv", "val2.csv"]
  }
}
```
**How it works:**

- Pairs each training dataset with its corresponding validation dataset
- Processes pairs in parallel for efficiency
- Training data is further split 80/20 for train/test

### 2. single_dataset_split Mode

**When to use:**

- You have single dataset files that need to be split
- You want the script to handle train/val/test splitting
- You prefer random splitting over predetermined splits

**Required configuration:**

```json
{
  "runtime_params": {
    "processing_mode": "single_dataset_split",
    "base_directory": "/path/to/data",
    "training_datasets": ["dataset1.csv", "dataset2.csv"],
    "dataset_size": 1000
  }
}
```
**How it works:**

- Takes dataset_size samples for validation
- Takes another dataset_size samples for training+testing
- Splits training data 80/20 for final train/test sets

### 3. batch_noise_folders Mode

**When to use:**

- You have multiple experiment folders (ending with 'Noise')
- Each folder contains multiple CSV files to process
- You want to batch process an entire experimental setup

**Required configuration:**

```json

{
  "runtime_params": {
    "processing_mode": "batch_noise_folders",
    "base_directory": "/path/to/experiments",
    "dataset_size": 1000
  }
}
```

**How it works:**

- Scans base_directory for folders ending with 'Noise'
- Processes every CSV file in each folder
- Uses single_dataset_split logic for each file

Examples
**Running with separate validation:**
```bash
python pysr_runner.py --config separate_val_config.json
```

**Running single dataset with override:**


```bash
python pysr_runner.py --config single_dataset_config.json --runtime 3600 --debug
```

**Running batch processing:**
```bash
python pysr_runner.py --config batch_config.json --mode batch_noise_folders
```