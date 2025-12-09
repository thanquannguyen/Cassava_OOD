import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from models.mobilenet import CassavaMobileNet
import os
import sys
import torch.nn.functional as F
import pandas as pd

# Add src to path to import utils
sys.path.append(os.path.dirname(__file__))
from utils.dataset import CassavaDataset

# Configuration
ID_DATA_DIR = "data/cassava" 
OOD_DATA_PATH = "data/flowers102"
MODEL_PATH = "checkpoints/best_model.pth" # Updated from cassava_mobilenet_v3.pth
SAVE_PATH = "checkpoints/cassava_mobilenet_v3_robust.pth"
BATCH_SIZE = 16 # Reduced from 32 to fix OOM on 4GB GPU
LR = 0.0001
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Energy Loss Hyperparameters
M_IN = -5.0   # Want ID score > -5
M_OUT = -10.0 # Want OOD score < -10
LAMBDA_ENERGY = 0.1

def log_sum_exp(x, axis=1):
    return torch.logsumexp(x, dim=axis)

def main():
    print(f"Using device: {DEVICE}")

    # 1. Prepare Datasets
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ID Dataset (Cassava) using custom CassavaDataset
    try:
        # Load the training split CSV
        train_df = pd.read_csv(os.path.join(ID_DATA_DIR, 'train_split.csv'))
        train_images_dir = os.path.join(ID_DATA_DIR, 'train_images')
        
        train_dataset = CassavaDataset(train_df, train_images_dir, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        print(f"ID Dataset (Cassava): {len(train_dataset)} images loaded from {train_images_dir}.")
    except Exception as e:
        print(f"Error loading Cassava dataset: {e}")
        # Check if files exist
        if not os.path.exists(os.path.join(ID_DATA_DIR, 'train_split.csv')):
             print("TIP: train_split.csv missing. Run src/utils/split_data.py?")
        sys.exit(1)

    # OOD Dataset (Flowers102) - Acting as Noise/outliers
    os.makedirs(OOD_DATA_PATH, exist_ok=True)
    try:
        ood_dataset = datasets.Flowers102(root=OOD_DATA_PATH, split='train', download=True, transform=transform)
        print(f"OOD Dataset (Flowers102): {len(ood_dataset)} images.")
    except Exception as e:
        print(f"Flowers102 failed: {e}. Fallback to CIFAR100.")
        ood_dataset = datasets.CIFAR100(root='data/cifar100', download=True, transform=transform)
        print(f"OOD Dataset (CIFAR100): {len(ood_dataset)} images.")
    
    ood_loader = DataLoader(ood_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 2. Load Model
    model = CassavaMobileNet(num_classes=5, pretrained=False).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing weights from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print("Warning: Pretrained model not found at {MODEL_PATH}. Training from scratch (NOT RECOMMENDED).")
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # 3. Training Loop
    model.train()
    print("Starting robustness training...")
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        running_ce = 0.0
        running_energy = 0.0
        
        ood_iter = iter(ood_loader)
        
        # Limit to partial dataset per epoch to save time? No, full epoch is safer.
        for i, (inputs_id, labels_id) in enumerate(train_loader):
            inputs_id, labels_id = inputs_id.to(DEVICE), labels_id.to(DEVICE)
            
            # Get OOD Batch
            try:
                inputs_ood, _ = next(ood_iter)
            except StopIteration:
                ood_iter = iter(ood_loader)
                inputs_ood, _ = next(ood_iter)
            
            # OOD batch size might differ if end of loader, resize if needed or just use
            inputs_ood = inputs_ood.to(DEVICE)
            if inputs_ood.size(0) != inputs_id.size(0):
                 # Just skip incomplete batches for simplicity in loss pairing if needed, 
                 # but here losses are means, so it's fine.
                 pass

            # Forward ID
            optimizer.zero_grad()
            logits_id = model(inputs_id)
            loss_ce = criterion(logits_id, labels_id)
            
            # Forward OOD
            logits_ood = model(inputs_ood)
            
            # Energy Scores
            energy_id = log_sum_exp(logits_id)
            energy_ood = log_sum_exp(logits_ood)
            
            # Margin Loss
            loss_margin = 0.1 * (
                torch.pow(F.relu(M_IN - energy_id), 2).mean() +
                torch.pow(F.relu(energy_ood - M_OUT), 2).mean()
            )
            
            loss = loss_ce + LAMBDA_ENERGY * loss_margin
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_ce += loss_ce.item()
            running_energy += loss_margin.item()
            
            if i % 20 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} [{i}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} (CE: {loss_ce.item():.4f}, OE: {loss_margin.item():.4f}) "
                      f"E_in: {energy_id.mean():.2f}, E_out: {energy_ood.mean():.2f}")

        print(f"Epoch {epoch+1} Complete. Avg Loss: {running_loss/len(train_loader):.4f}")
        
        # Save Checkpoint
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()
