import pandas as pd
from train import train_model
from src.data_utils import load_mnist
import os

def run_experiments(experiments, csv_name="results.csv"):
    print("Loading data...")
    X_train, X_val, X_test = load_mnist('data')
    
    results = []
    
    for config in experiments:
        metrics = train_model(config, X_train, X_val, X_test)
        results.append(metrics)
        
        # Save intermediate results
        df = pd.DataFrame(results)
        df.to_csv(f'results/metrics/{csv_name}', index=False)
        print(f"Finished {config['name']}. Test MSE: {metrics['test_mse']:.4f}\n")
    return df
