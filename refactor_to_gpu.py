import os

backend_code = """
try:
    import cupy as np
    GPU_AVAILABLE = True
except ImportError:
    import numpy as np
    GPU_AVAILABLE = False
"""

with open("C:\\Users\\PAUL\\Documents\\PRML\\autoencoder_project\\src\\backend.py", "w") as f:
    f.write(backend_code)

def replace_import(filepath):
    with open(filepath, "r") as f:
        content = f.read()
    
    # Replace direct numpy import with backend
    content = content.replace("import numpy as np", "from .backend import np")
    
    with open(filepath, "w") as f:
        f.write(content)

# Update core math files
for file in ["layers.py", "activations.py", "losses.py", "optimizers.py", "metrics.py"]:
    replace_import(f"C:\\Users\\PAUL\\Documents\\PRML\\autoencoder_project\\src\\{file}")

# Update train.py
with open("C:\\Users\\PAUL\\Documents\\PRML\\autoencoder_project\\train.py", "r") as f:
    train_content = f.read()

train_content = train_content.replace("import numpy as np", "from src.backend import np, GPU_AVAILABLE")

# Convert data to GPU arrays before training
gpu_conversion = """
    # Build model
    model = Autoencoder(
"""
gpu_conversion_replacement = """
    if GPU_AVAILABLE:
        print("Using GPU for training...")
        X_train = np.asarray(X_train)
        X_val = np.asarray(X_val)
        X_test = np.asarray(X_test)
    else:
        print("Using CPU for training...")

    # Build model
    model = Autoencoder(
"""
train_content = train_content.replace(gpu_conversion, gpu_conversion_replacement)

# Make sure train_losses and val_losses convert back to CPU float before appending
train_content = train_content.replace("train_losses.append(train_loss)", "train_losses.append(float(train_loss))")
train_content = train_content.replace("val_losses.append(val_loss)", "val_losses.append(float(val_loss))")
train_content = train_content.replace("test_mse = calculate_mse(X_test, test_pred)", "test_mse = float(calculate_mse(X_test, test_pred))")

with open("C:\\Users\\PAUL\\Documents\\PRML\\autoencoder_project\\train.py", "w") as f:
    f.write(train_content)

# Update visualization to handle CuPy arrays
with open("C:\\Users\\PAUL\\Documents\\PRML\\autoencoder_project\\src\\visualization.py", "r") as f:
    vis_content = f.read()

vis_fix = """def visualize_reconstructions(original, reconstructed, save_path, n=10, title="Reconstructions"):
    # Convert from CuPy to NumPy if necessary for matplotlib
    if hasattr(original, 'get'):
        original = original.get()
    if hasattr(reconstructed, 'get'):
        reconstructed = reconstructed.get()
"""
vis_content = vis_content.replace("def visualize_reconstructions(original, reconstructed, save_path, n=10, title=\"Reconstructions\"):", vis_fix)

with open("C:\\Users\\PAUL\\Documents\\PRML\\autoencoder_project\\src\\visualization.py", "w") as f:
    f.write(vis_content)
