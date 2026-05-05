import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_training_data(n_samples=5000):
    """
    Generates a dataset for training effectiveness based on intensity and duration.
    The logic reflects polarized training principles, leading to an XOR-style distribution:
    - Low Intensity / Low Duration -> 0 (No adaptation)
    - Low Intensity / High Duration -> 1 (Endurance Base)
    - High Intensity / Low Duration -> 1 (HIIT / VO2 Max)
    - High Intensity / High Duration -> 0 (Overtraining / Burnout)
    """
    np.random.seed(42)
    
    # Generate evenly spread data over realistic real-world ranges
    # Intensity: 60% to 100% of Max HR
    intensity_hr_pct = np.random.uniform(0.60, 1.00, n_samples)
    
    # Duration: 0.2 to 3.0 hours (12 minutes to 3 hours)
    duration_hours = np.random.uniform(0.2, 3.0, n_samples)
    
    # Normalize inputs to a [0, 1] range
    i_norm = (intensity_hr_pct - 0.60) / 0.40
    d_norm = (duration_hours - 0.2) / 2.8
    
    # Goldilocks / XOR Hybrid logic:
    # 1. Start with a baseline to make the data "more red" (generally effective)
    baseline = 0.45
    
    # 2. Add an XOR component but dampen it so it doesn't dominate the corners
    # Increased coefficient (from 0.3 to 0.45) to make the corners more "red"
    xor_component = 0.45 * (i_norm * (1 - d_norm) + d_norm * (1 - i_norm))
    
    # 3. Add a "Goldilocks" center boost (moderate intensity, moderate duration)
    center_boost = 0.4 * np.exp(-((i_norm - 0.5)**2 + (d_norm - 0.5)**2) / 0.15)
    
    effectiveness_raw = baseline + xor_component + center_boost
    
    # Add real-world variance/noise
    noise = np.random.normal(0, 0.04, n_samples)
    effectiveness = effectiveness_raw + noise
    
    # Rescale to strictly go from 0 to 1
    e_min = effectiveness.min()
    e_max = effectiveness.max()
    effectiveness = (effectiveness - e_min) / (e_max - e_min)
    
    return pd.DataFrame({
        'Intensity_MaxHR_Pct': round(pd.Series(intensity_hr_pct), 3),
        'Duration_Hours': round(pd.Series(duration_hours), 3),
        'Training_Effectiveness': round(pd.Series(effectiveness), 4)
    })

def plot_effectiveness(df, save_path='training_effectiveness_plot.png'):
    """
    Creates a 2D scatter plot mapping Duration (x) and Intensity (y) with Effectiveness as color.
    """
    plt.figure(figsize=(11, 8))
    
    # Create the scatter plot
    scatter = plt.scatter(
        df['Duration_Hours'], 
        df['Intensity_MaxHR_Pct'], 
        c=df['Training_Effectiveness'], 
        cmap='coolwarm',  
        alpha=0.8,
        s=30,
        linewidth=0,
        vmin=0.0,
        vmax=1.0,
    )
    
    # Add title
    plt.title('Training Effectiveness by Intensity and Duration', fontsize=16)

    plt.xlabel('Training Duration (Hours/Day)', fontsize=12)
    plt.ylabel('Training Intensity (Average HR % of Max)', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Training Effectiveness (0 to 1)', fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Export and show logic
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully as '{save_path}'")
    plt.show()

if __name__ == "__main__":
    print("Generating dataset with 5000 uniform samples...")
    df = generate_training_data(5000)
    
    csv_path = "training_effectiveness_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"Dataset securely saved to: {csv_path}")
    
    print("\nFirst 5 entries:")
    print(df.head())
    
    print("\nGenerating and displaying 2D visualization...")
    plot_effectiveness(df)
