"""
Training Effectiveness Neural Network

How to run:
1. Ensure the dataset exists by running:
   python3 training_effectiveness_dataset.py
2. Run this neural network training script:
   python3 train_effectiveness_nn.py

How to change parameters:
Edit the HYPERPARAMETERS section below to adjust the model architecture and training settings.
"""
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# --- HYPERPARAMETERS ---
EPOCHS = 10000
BATCH_SIZE = 100
LEARNING_RATE = 0.5
HIDDEN_LAYER_SIZE = 5   # Change this to modify the number of neurons in the hidden layer
PLOT_HIDDEN_NEURONS = True  # Set to True to generate individual neuron boundary plots
# -----------------------

def plot_training_stats(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, 'b-', linewidth=1.5, label='Training Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average BCE Loss', fontsize=12)
    plt.title('Training Loss Over Time (Mini-Batch)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    filename = 'effectiveness_training_stats.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training loss plot saved to: {filename}")

def plot_decision_boundary(X_original, y_original, X_norm_min, X_norm_max, 
                           W1, b1, W2, b2,
                           save_path='effectiveness_decision_boundary.png',
                           neuron='output'):
    
    # Create grid over input space (original scale)
    dur_min, dur_max = X_norm_min[0], X_norm_max[0]
    int_min, int_max = X_norm_min[1], X_norm_max[1]
    
    xs = np.linspace(dur_min, dur_max, 100)
    ys = np.linspace(int_min, int_max, 100)
    xx, yy = np.meshgrid(xs, ys)
    
    # Normalize grid points for NN prediction
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    Z = []
    for d, i in grid_points:
        # Normalize inputs exactly like training data
        d_norm = (d - dur_min) / (dur_max - dur_min) if dur_max > dur_min else 0
        i_norm = (i - int_min) / (int_max - int_min) if int_max > int_min else 0
        
        # Forward pass
        x_in = np.array([[d_norm, i_norm]])
        z1 = x_in @ W1 + b1
        h1 = 1 / (1 + np.exp(-np.clip(z1, -20, 20)))
        
        if neuron == 'output':
            z2 = h1 @ W2 + b2
            y_pred = 1 / (1 + np.exp(-np.clip(z2, -20, 20)))
            val = y_pred[0, 0]
        else:
            val = h1[0, neuron]
            
        Z.append(val)
        
    Z = np.array(Z).reshape(xx.shape)

    plt.figure(figsize=(11, 8))
    
    # Colored confidence map (activation value) from the NN
    norm = Normalize(vmin=0.0, vmax=1.0)
    contour = plt.contourf(
        xx, yy, Z,
        levels=np.linspace(0, 1, 51),
        cmap="coolwarm",
        norm=norm,
        alpha=0.8,
        zorder=1
    )
    
    # Draw the decision boundary at y=0.5 (could be multiple contours)
    plt.contour(
        xx, yy, Z,
        levels=[0.5],
        colors="k",
        linewidths=[2],
        linestyles=['-'],
        zorder=2
    )
    
    # Scatter original dataset (alpha=0.8 to match the dataset plot exactly)
    scatter = plt.scatter(
        X_original[:, 0], 
        X_original[:, 1], 
        c=y_original, 
        cmap='coolwarm',  
        alpha=0.8,
        s=30,
        linewidth=0.5,
        edgecolors='black',
        zorder=5,
        vmin=0.0,
        vmax=1.0
    )
    
    # Colorbar from scatter (matches the dataset plot's colorbar exactly)
    cbar1 = plt.colorbar(scatter, pad=0.02)
    cbar1.set_ticks(np.linspace(0, 1, 6))
    if neuron == 'output':
        cbar1.set_label("NN Predicted Effectiveness (0 to 1)", fontsize=12)
        title = f"NN Decision Boundary over Effectiveness Dataset (Hidden Layer Size = {HIDDEN_LAYER_SIZE})"
    else:
        cbar1.set_label(f"Hidden Neuron {neuron+1} Activation (0 to 1)", fontsize=12)
        title = f"Hidden Neuron {neuron+1} Decision Boundary (Hidden Layer Size = {HIDDEN_LAYER_SIZE})"
        
    plt.title(title, fontsize=16)
    plt.xlabel('Training Duration (Hours/Day)', fontsize=12)
    plt.ylabel('Training Intensity (Average HR % of Max)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Decision boundary plot saved to: {save_path}")


def train():
    # 1. Load data
    try:
        df = pd.read_csv('training_effectiveness_dataset.csv')
    except FileNotFoundError:
        print("Dataset not found! Please run 'python training_effectiveness_dataset.py' first.")
        return
        
    # Extract features and targets
    X_raw = df[['Duration_Hours', 'Intensity_MaxHR_Pct']].values
    y_raw = df['Training_Effectiveness'].values
    
    # 2. Normalize inputs to [0, 1] for stable NN training
    X_min = X_raw.min(axis=0)
    X_max = X_raw.max(axis=0)
    X_norm = (X_raw - X_min) / (X_max - X_min)
    
    N = len(X_norm)
    print(f"Loaded {N} samples.")
    
    # 3. Initialize weights (Randomly)
    np.random.seed(42)
    
    # Layer 1 weights and biases
    W1 = np.random.randn(2, HIDDEN_LAYER_SIZE)
    b1 = np.random.randn(HIDDEN_LAYER_SIZE)
    
    # Layer 2 weights and biases
    W2 = np.random.randn(HIDDEN_LAYER_SIZE, 1)
    b2 = np.random.randn(1)
    
    epoch_losses = []
    
    print(f"Starting training:")
    print(f" - Epochs: {EPOCHS}")
    print(f" - Mini-batch size: {BATCH_SIZE}")
    print(f" - Hidden Layer Size: {HIDDEN_LAYER_SIZE}")
    print(f" - Learning Rate: {LEARNING_RATE}")
    
    for epoch in range(EPOCHS):
        # Shuffle data for each epoch
        indices = np.random.permutation(N)
        X_shuffled = X_norm[indices]
        y_shuffled = y_raw[indices]
        
        batch_losses = []
        
        # Mini-batch gradient descent vectorized
        for start_idx in range(0, N, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, N)
            current_batch_size = end_idx - start_idx
            
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx].reshape(-1, 1)
            
            # Forward pass
            Z1 = X_batch @ W1 + b1
            H1 = 1 / (1 + np.exp(-np.clip(Z1, -20, 20)))
            
            Z2 = H1 @ W2 + b2
            Y_pred = 1 / (1 + np.exp(-np.clip(Z2, -20, 20)))
            
            # Loss computation
            eps = 1e-9
            Y_pred_clipped = np.clip(Y_pred, eps, 1 - eps)
            total_batch_loss = -np.sum(y_batch * np.log(Y_pred_clipped) + (1 - y_batch) * np.log(1 - Y_pred_clipped))
            
            # Backward pass
            delta2 = Y_pred - y_batch
            
            dW2 = H1.T @ delta2
            db2 = np.sum(delta2, axis=0)
            
            delta1 = (delta2 @ W2.T) * H1 * (1 - H1)
            
            dW1 = X_batch.T @ delta1
            db1 = np.sum(delta1, axis=0)
            
            # Average gradients and update weights
            W1 -= LEARNING_RATE * (dW1 / current_batch_size)
            b1 -= LEARNING_RATE * (db1 / current_batch_size)
            
            W2 -= LEARNING_RATE * (dW2 / current_batch_size)
            b2 -= LEARNING_RATE * (db2 / current_batch_size)
            
            batch_losses.append(total_batch_loss / current_batch_size)
            
        epoch_avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(epoch_avg_loss)
        

            
    print("\n" + "="*40)
    print("        FINAL TRAINING STATS        ")
    print("="*40)
    print(f"Total Epochs Run      : {EPOCHS}")
    print(f"Final NN Configuration: 2-{HIDDEN_LAYER_SIZE}-1")
    print(f"Final Train Loss (BCE): {epoch_losses[-1]:.4f}")
    
    print("-" * 40)
    print("Final Output Layer Weights (W2):")
    for i in range(HIDDEN_LAYER_SIZE):
        print(f"  w{i+1}->out : {W2[i, 0]:8.4f}")
    print(f"Final Output Layer Bias (b2)   : {b2[0]:8.4f}")
    print("="*40 + "\n")
    
    # 4. Plot loss curve
    plot_training_stats(epoch_losses)
    
    # 5. Plot decision boundary over the original data
    plot_decision_boundary(X_raw, y_raw, X_min, X_max, W1, b1, W2, b2)
    
    if PLOT_HIDDEN_NEURONS:
        for i in range(HIDDEN_LAYER_SIZE):
            plot_decision_boundary(
                X_raw, y_raw, X_min, X_max, W1, b1, W2, b2,
                save_path=f'effectiveness_decision_boundary_h{i+1}.png',
                neuron=i
            )

if __name__ == "__main__":
    train()
