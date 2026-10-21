import os
import time
from src.backend import np, GPU_AVAILABLE
import pandas as pd
from src.autoencoder import Autoencoder
from src.losses import MSE
from src.optimizers import get_optimizer
from src.data_utils import load_mnist
from src.metrics import calculate_mse, compression_ratio
from src.visualization import plot_loss_curves, visualize_reconstructions

def train_model(config, X_train, X_val, X_test):
    print(f"--- Running experiment: {config['name']} ---")
    
    # Set random seed
    np.random.seed(config.get('seed', 42))
    
    if GPU_AVAILABLE:
        print("Using GPU for training...")
        X_train = np.asarray(X_train)
        X_val = np.asarray(X_val)
        X_test = np.asarray(X_test)
    else:
        print("Using CPU for training...")

    # Build model
    model = Autoencoder(
        input_size=config['input_size'],
        encoder_layers=config['encoder'],
        latent_size=config['latent'],
        decoder_layers=config['decoder'],
        activation_function=config.get('activation', 'relu'),
        output_activation=config.get('output_activation', 'sigmoid'),
        regularization=config.get('regularization', None),
        regularization_strength=config.get('regularization_strength', 0.0)
    )
    
    loss_fn = MSE()
    optimizer = get_optimizer(config.get('optimizer', 'adam'), config.get('learning_rate', 0.001))
    
    batch_size = config.get('batch_size', 128)
    epochs = config.get('epochs', 50)
    
    train_losses = []
    val_losses = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        
        epoch_loss = 0
        batches = 0
        
        for i in range(0, X_train.shape[0], batch_size):
            batch_X = X_train_shuffled[i:i+batch_size]
            
            # Forward pass
            y_pred = model.forward(batch_X)
            
            # Add noise if denoising
            if config.get('denoising', False):
                noise_factor = config.get('noise_factor', 0.2)
                batch_X_noisy = batch_X + noise_factor * np.random.randn(*batch_X.shape)
                batch_X_noisy = np.clip(batch_X_noisy, 0., 1.)
                y_pred = model.forward(batch_X_noisy)
            
            loss = loss_fn.forward(y_pred, batch_X)
            
            # Backward pass
            dvalues = loss_fn.backward()
            model.backward(dvalues)
            
            # Update parameters
            model.update_params(optimizer)
            
            epoch_loss += loss
            batches += 1
            
        train_loss = epoch_loss / batches
        train_losses.append(float(train_loss))
        
        # Validation
        val_pred = model.forward(X_val)
        val_loss = loss_fn.forward(val_pred, X_val)
        val_losses.append(float(val_loss))
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
    training_time = time.time() - start_time
    
    # Test
    test_pred = model.forward(X_test)
    test_mse = float(calculate_mse(X_test, test_pred))
    
    # Save visualizations
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/reconstructions', exist_ok=True)
    
    plot_loss_curves(train_losses, val_losses, f"results/plots/{config['name']}_loss.png")
    
    if config.get('denoising', False):
        noise_factor = config.get('noise_factor', 0.2)
        X_test_noisy = X_test[:10] + noise_factor * np.random.randn(*X_test[:10].shape)
        X_test_noisy = np.clip(X_test_noisy, 0., 1.)
        test_pred_vis = model.forward(X_test_noisy)
        visualize_reconstructions(X_test_noisy, test_pred_vis, f"results/reconstructions/{config['name']}.png")
    else:
        visualize_reconstructions(X_test[:10], test_pred[:10], f"results/reconstructions/{config['name']}.png")
        
    metrics = {
        'experiment': config['name'],
        'architecture': f"{config['encoder']}->{config['latent']}->{config['decoder']}",
        'latent_size': config['latent'],
        'activation': config.get('activation', 'relu'),
        'learning_rate': config.get('learning_rate', 0.001),
        'batch_size': batch_size,
        'optimizer': config.get('optimizer', 'adam'),
        'parameters': model.count_parameters(),
        'training_time': training_time,
        'train_mse': train_losses[-1],
        'validation_mse': val_losses[-1],
        'test_mse': test_mse,
        'compression_ratio': compression_ratio(config['input_size'], config['latent'])
    }
    
    return metrics
