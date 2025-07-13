# lstm-btc-price
> BTC Price Predictor with LSTM + Short-term Calibration Bias (Optimized with GA)

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Structure](#structure)

## Overview
This framework is based on the paper available at https://doi.org/10.3390/app14188576, remade to predict BTC price. While the LSTM + STBC architecture is completely kept, the Genetic Algorithm has been modified in several procedures. The hyperparameters have not been fine-tuned, they are kept as it is in the paper. Fine-tuning is planned for the future roadmap. The data is taken from [CoinMarketCap](https://coinmarketcap.com/), it is kept raw for transparency.

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/tjkeat123/lstm-btc-price
cd lstm-btc-price
python3 -m venv venv # optional (recommended to use virtual env.)
pip install -r requirements.txt
```

## Usage

The project provides a command-line interface (CLI) for easy interaction:

### Basic Commands

```bash
# Get help on available commands
python3 cli.py --help

# Clean and prepare the raw data
python3 cli.py clean

# Train the LSTM model
python3 cli.py train

# Make predictions using the trained model
python3 cli.py predict
```

### Advanced Options

#### Training Options

```bash
# Train with custom parameters
python3 cli.py train --past-days 30 --future-days 1 --test-size 0.25 --epochs 50 --batch-size 64
```

#### Prediction Options

```bash
# Predict with custom parameters
python3 cli.py predict --model-path "models/lstm_paper.keras" --past-days 20 --calibrate --optimize --pop-size 150 --generations 25

# Predict without calibration
python3 cli.py predict --no-calibrate

# Predict with calibration but without optimization
python3 cli.py predict --calibrate --no-optimize
```

#### Output Options

```bash
# Predict without saving the plot and the results as a CSV 
python3 cli.py predict --no-save
```

## Structure

The project is organized as follows:

- `cli.py`: Command-line interface for all operations
- `data/`: Directory containing raw and processed data
  - `raw/`: Raw Bitcoin price data
  - `processed/`: Cleaned price data
- `models/`: Currently used models
- `models_backup/`: Backup of trained models with timestamps
- `results/`: Results from model predictions with timestamps
- `src/`: Source code directory
  - `bias/`: Bias compensation algorithms (STBC)
  - `data/`: Data loading, cleaning, and preprocessing
  - `features/`: Technical indicators calculation
  - `models/`: LSTM model definition and training
  - `optim/`: Optimization algorithms (SSGA)
  - `utils/`: Utility functions (plotting, reproducibility)