import math
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# python3 xor_nn.py
# ==========================================
# NETWORK PARAMETERS
# ==========================================
EPOCHS = 491
LEARNING_RATE = 1.0
VERBOSE = 0  # 0=minimal, 1=per-epoch loss, 2=full detail

# Set this to your final trained weights to skip training and just plot, 
# or set to None to train from fixed initial weights.
MANUAL_WEIGHTS = None
# MANUAL_WEIGHTS = (
#     (-5.038251, -4.936800, 1.066735),
#     (1.566796, 1.279671, -1.804769),
#     (-5.371513, -2.746245, 2.017824),
# )

# Fixed initial weights (always used for training)
FIXED_W1 = (-0.9, -0.5, 0.3)   # w11, w12, b1  - Hidden neuron 1
FIXED_W2 = (0.2, -0.1, 0.2)    # w21, w22, b2  - Hidden neuron 2
FIXED_W3 = (0.1, -0.3, 0.4)    # v1, v2, b3    - Output neuron

# Binary cross-entropy loss
def bce_loss(t, y, eps=1e-9):
    y = max(min(y, 1 - eps), eps)  # clamp to avoid log(0)
    return -(t * math.log(y) + (1 - t) * math.log(1 - y))

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Plot training statistics (loss over time)
def plot_training_stats(losses):
    """Create training loss visualization."""
    plt.figure(figsize=(10, 6))

    plt.plot(range(1, len(losses) + 1), losses, 'b-', linewidth=1.5, label='Training Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Binary Cross-Entropy Loss', fontsize=12)
    plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale to better see loss decrease
    plt.legend()

    plt.tight_layout()

    # Save the plot
    filename = 'xor_training_stats.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"Training loss plot saved to: {filename}")
    print(f"{'='*60}")
    plt.close()

def plot_decision_boundary(w_h1, w_h2, w_out, x_min=-0.5, x_max=1.5, y_min=-0.5, y_max=1.5, step=0.01, vmin=None, vmax=None, neuron='output'):
    """
    Visualize individual neuron decision boundaries.
    """
    w11, w12, b1 = w_h1
    w21, w22, b2 = w_h2
    v1, v2, b3 = w_out
    
    xs = np.arange(x_min, x_max, step)
    ys = np.arange(y_min, y_max, step)
    xx, yy = np.meshgrid(xs, ys)

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = []
    for x1, x2 in grid_points:
        h1 = sigmoid(w11 * x1 + w12 * x2 + b1)
        h2 = sigmoid(w21 * x1 + w22 * x2 + b2)
        if neuron == 'h1':
            y = h1
        elif neuron == 'h2':
            y = h2
        else:
            y = sigmoid(v1 * h1 + v2 * h2 + b3)
        Z.append(y)
    Z = np.array(Z).reshape(xx.shape)

    plt.figure(figsize=(7, 6))

    if vmin is not None and vmax is not None:
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = None
        vmin = np.min(Z)
        vmax = np.max(Z)
    
    contour = plt.contourf(
        xx,
        yy,
        Z,
        levels=50,
        cmap="RdYlBu",
        alpha=0.8,
        norm=norm,
    )
    
    cbar = plt.colorbar(contour)
    if neuron == 'h1':
        cbar_label = "Activation σ(z1)"
    elif neuron == 'h2':
        cbar_label = "Activation σ(z2)"
    else:
        cbar_label = "Output activation σ(z3)"
    cbar.set_label(cbar_label, fontsize=11)

    plt.contour(
        xx,
        yy,
        Z,
        levels=[0.5],
        colors="k",
        linewidths=2,
    )

    xor_inputs = np.array([[1, 1], [0, 0], [1, 0], [0, 1]])
    xor_targets = np.array([0, 0, 1, 1])
    colors = ["red" if t == 0 else "blue" for t in xor_targets]
    plt.scatter(
        xor_inputs[:, 0],
        xor_inputs[:, 1],
        c=colors,
        edgecolors="k",
        s=120,
        linewidths=1.5,
        label="XOR samples",
    )

    for (x1, x2), t in zip(xor_inputs, xor_targets):
        plt.text(
            x1 + 0.02,
            x2 + 0.02,
            str(t),
            fontsize=10,
            color="black",
            weight="bold",
        )

    plt.xlabel("Input x1", fontsize=12)
    plt.ylabel("Input x2", fontsize=12)
    
    if neuron == 'h1':
        title = "Hidden Neuron 1 Decision Boundary and Confidence"
        filename = "xor_decision_boundary_h1.png"
    elif neuron == 'h2':
        title = "Hidden Neuron 2 Decision Boundary and Confidence"
        filename = "xor_decision_boundary_h2.png"
    else:
        title = "XOR Decision Boundary and Output Confidence"
        filename = "xor_decision_boundary.png"

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"\n{'='*60}")
    print(f"Decision boundary plot saved to: {filename}")
    print(f"{'='*60}")
    plt.close()

# XOR dataset
X = [[1, 1], [0, 0], [1, 0], [0, 1]]
T = [0, 0, 1, 1]

def visualize_with_manual_weights():
    w_h1, w_h2, w_out = MANUAL_WEIGHTS

    print("=" * 60)
    print("Using MANUAL_WEIGHTS for visualization (no training).")
    print("=" * 60)
    print(f"  w11 = {w_h1[0]:.6f}, w12 = {w_h1[1]:.6f}, b1 = {w_h1[2]:.6f}")
    print(f"  w21 = {w_h2[0]:.6f}, w22 = {w_h2[1]:.6f}, b2 = {w_h2[2]:.6f}")
    print(f"  v1  = {w_out[0]:.6f}, v2  = {w_out[1]:.6f}, b3 = {w_out[2]:.6f}")

    print(f"\nPredictions with MANUAL_WEIGHTS:")
    for i in range(4):
        x1, x2 = X[i]
        h1 = sigmoid(w_h1[0] * x1 + w_h1[1] * x2 + w_h1[2])
        h2 = sigmoid(w_h2[0] * x1 + w_h2[1] * x2 + w_h2[2])
        y = sigmoid(w_out[0] * h1 + w_out[1] * h2 + w_out[2])
        print(f"  ({x1}, {x2}) -> {y:.6f} (target: {T[i]})")

    dummy_losses = [0.5]
    plot_training_stats(dummy_losses)
    plot_decision_boundary(w_h1, w_h2, w_out, vmin=0.0, vmax=1.0, neuron='output')
    plot_decision_boundary(w_h1, w_h2, w_out, vmin=0.0, vmax=1.0, neuron='h1')
    plot_decision_boundary(w_h1, w_h2, w_out, vmin=0.0, vmax=1.0, neuron='h2')

def train():
    w11, w12, b1 = FIXED_W1
    w21, w22, b2 = FIXED_W2
    v1, v2, b3 = FIXED_W3

    print("="*60)
    print("XOR Neural Network - 2-2-1 Architecture")
    print("="*60)
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Initial Weights (Fixed):")
    print(f"  Hidden 1: w11={w11:.6f}, w12={w12:.6f}, b1={b1:.6f}")
    print(f"  Hidden 2: w21={w21:.6f}, w22={w22:.6f}, b2={b2:.6f}")
    print(f"  Output:   v1={v1:.6f}, v2={v2:.6f}, b3={b3:.6f}")
    print("="*60)

    epoch_losses = []
    
    for epoch in range(EPOCHS):
        dw11_sum, dw12_sum, db1_sum = 0, 0, 0
        dw21_sum, dw22_sum, db2_sum = 0, 0, 0
        dv1_sum, dv2_sum, db3_sum = 0, 0, 0
        total_loss = 0

        for i in range(4):
            x1, x2 = X[i]
            t = T[i]

            # Forward
            z1 = w11*x1 + w12*x2 + b1
            h1 = sigmoid(z1)

            z2 = w21*x1 + w22*x2 + b2
            h2 = sigmoid(z2)

            z3 = v1*h1 + v2*h2 + b3
            y = sigmoid(z3)

            total_loss += bce_loss(t, y)

            # Backprop
            delta3 = y - t
            dv1 = delta3 * h1
            dv2 = delta3 * h2
            db3 = delta3

            delta1 = delta3 * v1 * h1 * (1 - h1)
            delta2 = delta3 * v2 * h2 * (1 - h2)

            dw11 = delta1 * x1
            dw12 = delta1 * x2
            db1 = delta1

            dw21 = delta2 * x1
            dw22 = delta2 * x2
            db2 = delta2

            dw11_sum += dw11; dw12_sum += dw12; db1_sum += db1
            dw21_sum += dw21; dw22_sum += dw22; db2_sum += db2
            dv1_sum += dv1; dv2_sum += dv2; db3_sum += db3

        # Update
        avg_loss = total_loss / 4
        epoch_losses.append(avg_loss)

        w11 -= LEARNING_RATE * (dw11_sum / 4)
        w12 -= LEARNING_RATE * (dw12_sum / 4)
        b1  -= LEARNING_RATE * (db1_sum / 4)
        
        w21 -= LEARNING_RATE * (dw21_sum / 4)
        w22 -= LEARNING_RATE * (dw22_sum / 4)
        b2  -= LEARNING_RATE * (db2_sum / 4)
        
        v1  -= LEARNING_RATE * (dv1_sum / 4)
        v2  -= LEARNING_RATE * (dv2_sum / 4)
        b3  -= LEARNING_RATE * (db3_sum / 4)

        if VERBOSE >= 1 and (epoch + 1) % max(1, (EPOCHS // 10)) == 0:
            print(f"Epoch {epoch+1:4d}: Avg Loss = {avg_loss:.6f}")

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Final Weights:")
    print(f"  w11 = {w11:.6f}, w12 = {w12:.6f}, b1 = {b1:.6f}")
    print(f"  w21 = {w21:.6f}, w22 = {w22:.6f}, b2 = {b2:.6f}")
    print(f"  v1  = {v1:.6f}, v2  = {v2:.6f}, b3 = {b3:.6f}")

    print(f"\nPredictions:")
    for i in range(4):
        x1, x2 = X[i]
        h1 = sigmoid(w11 * x1 + w12 * x2 + b1)
        h2 = sigmoid(w21 * x1 + w22 * x2 + b2)
        y = sigmoid(v1 * h1 + v2 * h2 + b3)
        print(f"  ({x1}, {x2}) -> {y:.6f} (target: {T[i]})")

    plot_training_stats(epoch_losses)
    
    w_h1, w_h2, w_out = (w11, w12, b1), (w21, w22, b2), (v1, v2, b3)
    plot_decision_boundary(w_h1, w_h2, w_out, vmin=0.0, vmax=1.0, neuron='output')
    plot_decision_boundary(w_h1, w_h2, w_out, vmin=0.0, vmax=1.0, neuron='h1')
    plot_decision_boundary(w_h1, w_h2, w_out, vmin=0.0, vmax=1.0, neuron='h2')


if __name__ == "__main__":
    if MANUAL_WEIGHTS is not None:
        visualize_with_manual_weights()
    else:
        train()
