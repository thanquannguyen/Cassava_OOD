import time
import torch
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.mobilenet import CassavaMobileNet

def benchmark(model_path, device='cuda', num_runs=100, input_size=(1, 3, 224, 224)):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on {device}...")

    # Load Model
    model = CassavaMobileNet(num_classes=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Dummy Input
    input_tensor = torch.randn(input_size).to(device)

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    # Benchmark
    print(f"Running {num_runs} iterations...")
    latencies = []
    with torch.no_grad():
        for _ in tqdm(range(num_runs)):
            start_time = time.time()
            _ = model(input_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000) # ms

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    fps = 1000 / avg_latency

    print("\n--- Performance Results ---")
    print(f"Average Latency: {avg_latency:.2f} ms +/- {std_latency:.2f} ms")
    print(f"Average FPS: {fps:.2f}")
    
    return fps, avg_latency

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--runs', type=int, default=500)
    args = parser.parse_args()
    
    benchmark(args.model_path, num_runs=args.runs)
