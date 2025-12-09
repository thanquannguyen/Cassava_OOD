import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
import os
import sys
import pandas as pd
from PIL import Image
import torch.nn.functional as F

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Detect Kaggle Environment
ON_KAGGLE = os.path.exists('/kaggle/input')

if ON_KAGGLE:
    print("Detected Kaggle Environment!")
    ID_DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
    # User must upload their pretrained model as a dataset. 
    # Assumes it's in a dataset named 'cassava-pretrained' or similar. 
    # We will search for it or user updates this path.
    # Updated path based on user feedback:
    MODEL_PATH = "/kaggle/input/cassava-ood-mobilenetv3/pytorch/default/1/best_model.pth" 
    OOD_DATA_PATH = "./flowers102" # Download to working dir
    SAVE_DIR = "/kaggle/working"
else:
    print("Detected Local Environment")
    ID_DATA_DIR = "data/cassava"
    MODEL_PATH = "checkpoints/best_model.pth"
    OOD_DATA_PATH = "data/flowers102"
    SAVE_DIR = "checkpoints"

SAVE_PATH = os.path.join(SAVE_DIR, "cassava_mobilenet_v3_robust.pth")
BATCH_SIZE = 32 # Kaggle GPUs (P100/T4) can handle 32 easily
LR = 0.0001
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Energy Loss Params
M_IN = -5.0
M_OUT = -10.0
LAMBDA_ENERGY = 0.1

print(f"Device: {DEVICE}")
print(f"ID Data: {ID_DATA_DIR}")
print(f"Model Path: {MODEL_PATH}")
print(f"Output: {SAVE_PATH}")

# ==========================================
# 2. MODEL DEFINITION (Inline)
# ==========================================
class CassavaMobileNet(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(CassavaMobileNet, self).__init__()
        # Use MobileNetV3 Large to match original checkpoint
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_large(weights=weights)
        
        # Modify Head
        # The original classifier is a Sequential block. 
        # We replace the last linear layer [3].
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# ==========================================
# 3. DATASET DEFINITION (Inline)
# ==========================================
class CassavaDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            # print(f"Error loading {img_name}: {e}")
            image = Image.new('RGB', (224, 224))

        label = int(self.dataframe.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

def log_sum_exp(x, axis=1):
    return torch.logsumexp(x, dim=axis)

# ==========================================
# 4. MAIN TRAINING LOOP
# ==========================================
def main():
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Prepare ID Data ---
    # Check for train.csv (Kaggle structure)
    csv_path = os.path.join(ID_DATA_DIR, 'train.csv')
    img_dir = os.path.join(ID_DATA_DIR, 'train_images')
    
    if not os.path.exists(csv_path):
        # Maybe using the local split?
        csv_path = os.path.join(ID_DATA_DIR, 'train_split.csv')
        
    print(f"Loading ID data from {csv_path}...")
    try:
        train_df = pd.read_csv(csv_path)
        train_dataset = CassavaDataset(train_df, img_dir, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        print(f"-> Local/Kaggle Cassava loaded: {len(train_dataset)} images.")
    except Exception as e:
        print(f"Failed to load ID dataset: {e}")
        return

    # --- Prepare OOD Data ---
    print("Loading OOD data (Flowers102)...")
    os.makedirs(OOD_DATA_PATH, exist_ok=True)
    try:
        ood_dataset = datasets.Flowers102(root=OOD_DATA_PATH, split='train', download=True, transform=transform)
        print(f"-> Flowers102 loaded: {len(ood_dataset)} images.")
    except Exception as e:
        print(f"Flowers102 failed: {e}. Fallback to CIFAR100.")
        ood_dataset = datasets.CIFAR100(root='./cifar100', download=True, transform=transform)
        print(f"-> CIFAR100 loaded: {len(ood_dataset)} images.")

    ood_loader = DataLoader(ood_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # --- Load Model ---
    model = CassavaMobileNet(num_classes=5, pretrained=False).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        print(f"Loading pre-trained weights from {MODEL_PATH}")
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Make sure you uploaded your best_model.pth correctly!")
            # return # Don't exit, might want to train from scratch if intended
    else:
        print(f"Warning: {MODEL_PATH} not found. Training from scratch/ImageNet.")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # --- Training ---
    model.train()
    print("Starting Training...")
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        running_ce = 0.0
        running_energy = 0.0
        
        ood_iter = iter(ood_loader)
        
        for i, (inputs_id, labels_id) in enumerate(train_loader):
            inputs_id, labels_id = inputs_id.to(DEVICE), labels_id.to(DEVICE)
            
            # Get OOD Batch
            try:
                inputs_ood, _ = next(ood_iter)
            except StopIteration:
                ood_iter = iter(ood_loader)
                inputs_ood, _ = next(ood_iter)
            
            inputs_ood = inputs_ood.to(DEVICE)
            
            # Forward
            optimizer.zero_grad()
            logits_id = model(inputs_id)
            loss_ce = criterion(logits_id, labels_id)
            
            logits_ood = model(inputs_ood)
            
            # Energy Loss
            energy_id = log_sum_exp(logits_id)
            energy_ood = log_sum_exp(logits_ood)
            
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
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1} [{i}/{len(train_loader)}] "
                      f"L: {loss.item():.3f} (CE:{loss_ce.item():.3f} OE:{loss_margin.item():.3f}) "
                      f"E_in:{energy_id.mean():.1f} E_out:{energy_ood.mean():.1f}")
                      
        print(f"Epoch {epoch+1} Complete. Avg Loss: {running_loss/len(train_loader):.4f}")
        
    # Validating on "Foot" logic? Not possible on Kaggle easily without uploading that specific image.
    # Just save model.
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"SUCCESS! Model saved to: {SAVE_PATH}")
    print("Download this file and put it in your 'checkpoints' folder locally.")

if __name__ == "__main__":
    main()
