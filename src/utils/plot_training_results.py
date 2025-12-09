import matplotlib.pyplot as plt
import re

# Raw Logs from User's Kaggle Session
raw_logs = """
Epoch 1 [600/669] L: 0.142 E_in:-1.0 E_out:-11.2 (Avg Loss: 0.5313)
Epoch 2 [600/669] L: 0.182 E_in:0.7 E_out:-11.5 (Avg Loss: 0.2717)
Epoch 3 [600/669] L: 0.203 E_in:-2.1 E_out:-11.2 (Avg Loss: 0.2104)
Epoch 4 [600/669] L: 0.323 E_in:-0.2 E_out:-12.6 (Avg Loss: 0.1521)
Epoch 5 [600/669] L: 0.102 E_in:-1.3 E_out:-13.2 (Avg Loss: 0.0934)
Epoch 6 [600/669] L: 0.100 E_in:-0.2 E_out:-12.9 (Avg Loss: 0.0561)
Epoch 7 [600/669] L: 0.059 E_in:2.1 E_out:-13.5 (Avg Loss: 0.0353)
Epoch 8 [600/669] L: 0.005 E_in:1.8 E_out:-14.0 (Avg Loss: 0.0292)
Epoch 9 [600/669] L: 0.007 E_in:1.6 E_out:-15.1 (Avg Loss: 0.0226)
Epoch 10 [600/669] L: 0.003 E_in:3.9 E_out:-14.5 (Avg Loss: 0.0245)
"""

# Manual Data Extraction (approximate from end of epochs)
epochs = list(range(1, 11))
loss = [0.5313, 0.2717, 0.2104, 0.1521, 0.0934, 0.0561, 0.0353, 0.0292, 0.0226, 0.0245]

# Energy values (Taken from the last step of each epoch for trend)
# In Training: E_in increases (target > -5), E_out decreases (target < -10)
e_in = [-1.0, 0.7, -2.1, -0.2, -1.3, -0.2, 2.1, 1.8, 1.6, 3.9]
e_out = [-11.2, -11.5, -11.2, -12.6, -13.2, -12.9, -13.5, -14.0, -15.1, -14.5]

def plot_energy():
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, e_in, 'g-o', label='ID Energy (Cassava)', linewidth=2)
    plt.plot(epochs, e_out, 'r-o', label='OOD Energy (Noise)', linewidth=2)
    
    # Draw Training Targets
    plt.axhline(y=-5, color='g', linestyle='--', alpha=0.3, label='Target ID Bound (-5)')
    plt.axhline(y=-10, color='r', linestyle='--', alpha=0.3, label='Target OOD Bound (-10)')
    
    plt.title('Energy Score Separation during Training\n(Goal: Maximize Gap)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Free Energy (LogSumExp)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    output_path = 'src/utils/training_energy.png'
    plt.savefig(output_path)
    print(f"Saved {output_path}")

def plot_loss():
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'b-o', label='Total Loss', linewidth=2)
    plt.title('Training Loss Convergence', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    output_path = 'src/utils/training_loss.png'
    plt.savefig(output_path)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    plot_energy()
    plot_loss()
