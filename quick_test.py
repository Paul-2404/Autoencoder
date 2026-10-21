import sys
import os
from run_experiments import run_experiments

base_config = {
    "input_size": 784,
    "encoder": [128, 64],
    "latent": 32,
    "decoder": [64, 128],
    "activation": "relu",
    "output_activation": "sigmoid",
    "learning_rate": 0.001,
    "batch_size": 128,
    "epochs": 1,  # Only 1 epoch for quick testing
    "optimizer": "adam"
}

experiments = [
    {**base_config, "name": "quick_test"}
]

if __name__ == "__main__":
    run_experiments(experiments, "quick_test_results.csv")
