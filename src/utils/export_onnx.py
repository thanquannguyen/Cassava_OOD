import torch
import torch.onnx
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.mobilenet import CassavaMobileNet

def export_to_onnx(model_path, output_path, input_size=224):
    device = torch.device("cpu")
    model = CassavaMobileNet(num_classes=5).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found.")
        return

    model.eval()
    
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        verbose=False,
        input_names=['input'], 
        output_names=['output'],
        opset_version=11,
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Export complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--output_path', type=str, default='checkpoints/model.onnx')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    export_to_onnx(args.model_path, args.output_path)
