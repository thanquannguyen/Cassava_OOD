import os
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

def prepare_ood_data(root_dir, num_images=2000):
    """
    Downloads CIFAR-10 and saves a subset of images as JPEG files for OOD usage.
    """
    ood_dir = os.path.join(root_dir, 'ood')
    os.makedirs(ood_dir, exist_ok=True)
    
    print(f"Downloading CIFAR-10 to {root_dir}...")
    # Download CIFAR-10
    dataset = torchvision.datasets.CIFAR10(root=root_dir, train=False, download=True)
    
    print(f"Saving {num_images} images to {ood_dir}...")
    for i in tqdm(range(min(num_images, len(dataset)))):
        img, label = dataset[i]
        # Save as jpg
        img.save(os.path.join(ood_dir, f"ood_{i}.jpg"))
        
    print("OOD data preparation complete.")

if __name__ == "__main__":
    prepare_ood_data("c:/Data/Cassava_OOD/data")
