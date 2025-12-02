import os
import cv2
import numpy as np
import pandas as pd
import random

def create_dummy_data(base_dir):
    # 1. Create Directories
    cassava_dir = os.path.join(base_dir, 'data', 'cassava')
    ood_dir = os.path.join(base_dir, 'data', 'ood')
    train_images_dir = os.path.join(cassava_dir, 'train_images')
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(ood_dir, exist_ok=True)
    
    print(f"Created directories in {base_dir}")

    # 2. Create Dummy Cassava Images & CSV
    num_train = 50
    image_ids = []
    labels = []
    
    print("Generating dummy Cassava images...")
    for i in range(num_train):
        img_name = f"train_{i}.jpg"
        img_path = os.path.join(train_images_dir, img_name)
        
        # Create random image
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        # Add some text to make it look different
        cv2.putText(img, f"Class {i%5}", (50, 112), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(img_path, img)
        
        image_ids.append(img_name)
        labels.append(i % 5) # 5 classes
        
    df = pd.DataFrame({'image_id': image_ids, 'label': labels})
    df.to_csv(os.path.join(cassava_dir, 'train.csv'), index=False)
    print(f"Created train.csv with {num_train} samples.")

    # 3. Create Dummy OOD Images
    num_ood = 20
    print("Generating dummy OOD images...")
    for i in range(num_ood):
        img_name = f"ood_{i}.jpg"
        img_path = os.path.join(ood_dir, img_name)
        
        # Create random noise image (different distribution)
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.circle(img, (112, 112), 50, (0, 0, 255), -1)
        cv2.imwrite(img_path, img)
        
    print("Dummy data generation complete.")

if __name__ == "__main__":
    BASE_DIR = "c:/Data/Cassava_OOD"
    create_dummy_data(BASE_DIR)
