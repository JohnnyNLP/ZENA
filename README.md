# ZENA: Enhanced Anomaly-Based IDS for Smart Homes Automation Through Multilayer Artificial Neural Network

This is the official GitHub repository for the paper **"ZENA: Enhanced Anomaly-Based IDS for Smart Homes Automation Through Multilayer Artificial Neural Network"** published in the **Journal of Communications and Networks**.

## Overview

ZENA is an enhanced anomaly-based Intrusion Detection System (IDS) designed specifically for smart home automation environments. The system employs a multilayer artificial neural network to detect anomalous behaviors and potential security threats in smart home networks.

## Architecture

ZENA implements a three-layer neural network architecture:
- **Input Layer**: 30 features
- **First Hidden Layer**: 128 neurons with ReLU activation
- **Second Hidden Layer**: 32 neurons with ReLU activation
- **Output Layer**: Binary classification (sigmoid activation)

The model uses momentum-based gradient descent optimization and supports multiple weight initialization methods (Xavier, He, random, zero).

## Features

- Multi-layer artificial neural network for anomaly detection
- Early stopping mechanism to prevent overfitting
- Support for customizable hyperparameters
- Model parameter saving and loading functionality
- Comprehensive evaluation metrics (Accuracy, F1-Score)

## Requirements

- Python 3.x
- NumPy >= 1.19.0
- Pandas >= 1.0.0
- scikit-learn >= 0.23.0

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Train the ZENA model with default parameters:

```bash
python train.py
```

Train with custom parameters:

```bash
python train.py \
    --model_name ZENA \
    --input_size 30 \
    --hidden_size1 128 \
    --hidden_size2 32 \
    --learning_rate 0.03 \
    --iterations 100000 \
    --early_stopping_rounds 5000 \
    --tolerance 1e-4 \
    --init_method xavier \
    --momentum 0.9
```

### Using the Model in Python

```python
from model import ZENA
import numpy as np

# Initialize the model
model = ZENA(
    input_size=30,
    hidden_size1=128,
    hidden_size2=32,
    learning_rate=0.03,
    iterations=100000,
    early_stopping_rounds=5000,
    tolerance=1e-4,
    init_method='xavier',
    momentum=0.9
)

# Train the model
train_loss_history, val_loss_history = model.fit(X_train, Y_train, X_val, Y_val)

# Make predictions
predictions = model.predict(X_test)

# Save model parameters
model.save_params('params/ZENA_model.pkl')

# Load model parameters
model.load_params('params/ZENA_model.pkl')
```

## Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_size` | 30 | Number of input features |
| `hidden_size1` | 128 | Number of neurons in the first hidden layer |
| `hidden_size2` | 32 | Number of neurons in the second hidden layer |
| `learning_rate` | 0.03 | Learning rate for gradient descent |
| `iterations` | 100000 | Maximum number of training iterations |
| `early_stopping_rounds` | 5000 | Number of rounds for early stopping |
| `tolerance` | 1e-4 | Minimum improvement threshold for early stopping |
| `init_method` | 'xavier' | Weight initialization method (xavier, he, random, zero) |
| `momentum` | 0.9 | Momentum coefficient for gradient descent |

## Data Format

The model expects CSV files with the following format:
- Last column: Labels (0 for normal, 1 for anomaly)
- Other columns: Feature values (30 features by default)

The data will be automatically split into training (80%), validation (10%), and test (10%) sets.

## Project Structure

```
ZENA/
├── model.py              # ZENA model implementation
├── train.py              # Training script
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── ZENA_data.csv        # Training dataset
├── params/              # Saved model parameters
└── images/              # Figures and visualizations
```
