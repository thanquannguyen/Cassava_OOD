import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.mobilenet import CassavaMobileNet
from utils.dataset import CassavaDataset, OODDataset
from utils.metrics import calculate_auroc, calculate_fpr95, expected_calibration_error

def get_logits(model, loader, device):
    model.eval()
    logits_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Computing Logits"):
            images = images.to(device)
            outputs = model(images)
            logits_list.append(outputs.cpu().numpy())
            labels_list.append(labels.numpy())
            
    return np.concatenate(logits_list), np.concatenate(labels_list)

def temperature_scaling(logits, labels, device):
    # Find optimal T using NLL
    # Simple grid search or optimization
    print("Calibrating Temperature...")
    T = nn.Parameter(torch.ones(1, device=device) * 1.5) # Init > 1
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
    
    logits_torch = torch.from_numpy(logits).to(device)
    labels_torch = torch.from_numpy(labels).long().to(device)
    nll_criterion = nn.CrossEntropyLoss()
    
    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(logits_torch / T, labels_torch)
        loss.backward()
        return loss
    
    optimizer.step(eval)
    return T.item()

def compute_energy(logits, T):
    # E(x) = -T * logsumexp(f(x)/T)
    return -T * torch.logsumexp(torch.from_numpy(logits) / T, dim=1).numpy()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = CassavaMobileNet(num_classes=5).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Data Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load ID Data (Validation Set)
    val_df = pd.read_csv(os.path.join(args.data_dir, 'val_split.csv'))
    id_dataset = CassavaDataset(val_df, os.path.join(args.data_dir, 'train_images'), transform=transform)
    id_loader = DataLoader(id_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Load OOD Data
    ood_dataset = OODDataset(os.path.join(args.data_dir, '../ood'), transform=transform)
    ood_loader = DataLoader(ood_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # 1. Get Logits
    print("Processing ID Data...")
    id_logits, id_labels = get_logits(model, id_loader, device)
    
    print("Processing OOD Data...")
    ood_logits, _ = get_logits(model, ood_loader, device)
    
    # 2. Temperature Scaling (using ID Validation set)
    T_optimal = temperature_scaling(id_logits, id_labels, device)
    print(f"Optimal Temperature T: {T_optimal:.4f}")
    
    # 3. Compute Energy Scores
    id_energy = compute_energy(id_logits, T_optimal)
    ood_energy = compute_energy(ood_logits, T_optimal)
    
    # 4. Metrics
    auroc = calculate_auroc(id_energy, ood_energy)
    fpr95 = calculate_fpr95(id_energy, ood_energy)
    
    print(f"AUROC: {auroc:.4f}")
    print(f"FPR95: {fpr95:.4f}")
    
    # 5. Determine Threshold (at 95% TPR for ID)
    # ID scores are lower (more negative). We want to accept low energy.
    # Threshold should be the 95th percentile of ID energy? 
    # No, we want to KEEP 95% of ID.
    # So we cut off the top 5% of ID energy (highest energy = most likely OOD).
    threshold = np.percentile(id_energy, 95)
    print(f"Selected Energy Threshold (95% ID recall): {threshold:.4f}")
    
    # Save params
    with open(os.path.join(args.save_dir, 'calibration_params.txt'), 'w') as f:
        f.write(f"T={T_optimal}\n")
        f.write(f"Threshold={threshold}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/cassava')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    main(args)
