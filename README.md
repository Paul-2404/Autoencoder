# Basic Fully Connected Autoencoder Project

This project implements a basic fully connected (dense) autoencoder from scratch using NumPy, designed for the MNIST dataset. It includes an experimental framework to analyze the effects of latent space dimensions, network architectures, activations, learning rates, batch sizes, optimizers, and regularizations.

## Installation

```bash
pip install -r requirements.txt
```

## Running Experiments

You can run individual experiments:

```bash
python experiments/latent_dimension.py
python experiments/architecture.py
python experiments/activation.py
python experiments/learning_rate.py
```

Results (metrics in CSV format, loss plots, and reconstructed images) are saved in the `results/` directory.

## Implementation Details

The autoencoder components are built from scratch (NumPy):
- Forward and backward passes of Dense layers
- Optimizers (SGD, Adam)
- Losses (MSE, MAE)
- Activations (ReLU, Sigmoid, Tanh)
