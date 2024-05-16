# Data Preprocessing

This repository contains scripts to download, process, and prepare MIMIC-III waveform data for training machine learning models. The scripts are organized to follow a pipeline that ensures high-quality data for model training and validation.

## Scripts and Workflow

### 1. Download MIMIC-III Data

#### `download_mimic3.py`

This script downloads the MIMIC-III waveform database.

**Usage:**
```sh
python download_mimic3.py
```

### 2. Filter and Save Raw Signals

#### `records_handler.py`

This script processes the downloaded MIMIC-III records, filtering out only the records that have 'ART', 'PLETH', and 'II' signals with a duration of at least 20 seconds. The filtered records are saved as `.npz` files in the `raw_signals` folder. Additionally, it creates a `records_list.csv` file that lists all the saved records.

**Usage:**
```sh
python records_handler.py
```

### 3. Extract High-Quality Signals

#### `extract_high_qual_signals.py`

This script reads the `.npz` files from the `raw_signals` folder, extracts 60-second segments of high-quality signals, downsamples them from 125Hz to 100Hz, and saves them in the `high_qual_60s` folder.

**Usage:**
```sh
python extract_high_qual_signals.py
```

### 4. Split into Training and Validation Datasets

#### `split_train_valid_dataset.py`

This script consolidates the signals stored in the `high_qual_60s` folder, splits them into 30-second training and validation sets for both ART and PPG signals, and saves them in the `../Models/TrainDataSet/` directory.

**Usage:**
```sh
python split_train_valid_dataset.py
```

## Directory Structure

- `DataPreprocessing/`
  - `download_mimic3.py`
  - `records_handler.py`
  - `extract_high_qual_signals.py`
  - `split_train_valid_dataset.py`
  - `raw_signals/` (generated by `records_handler.py`)
  - `high_qual_60s/` (generated by `extract_high_qual_signals.py`)
- `Models/`
  - `TrainDataSet/` (generated by `split_train_valid_dataset.py`)

## Requirements

- Python 3.x
- Required Python packages are listed in the scripts and will be installed if not already present:
  - `numpy`
  - `wfdb`
  - `tqdm`
  - `scipy`
  - `sklearn`
  - `heartpy`
  - `pandas`

## Usage

1. Clone the repository.
2. Navigate to the `DataPreprocessing` directory.
3. Run the scripts in the order specified above.

```sh
git clone <repository_url>
cd DataPreprocessing
python download_mimic3.py
python records_handler.py
python extract_high_qual_signals.py
python split_train_valid_dataset.py
```

This workflow ensures that you have high-quality, preprocessed data ready for training and validation of machine learning models using the MIMIC-III dataset.
