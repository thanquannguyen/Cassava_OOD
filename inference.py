import cv2
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
import os
import sys
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from models.mobilenet import CassavaMobileNet

# Check if TensorRT is available (optional optimization)
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

class OODDetector:
    def __init__(self, model_path, calibration_file, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load Model
        self.model = CassavaMobileNet(num_classes=5).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Load Calibration Params
        self.T = 1.0
        self.threshold = -20.0
        if os.path.exists(calibration_file):
            with open(calibration_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('T='):
                        self.T = float(line.strip().split('=')[1])
                    elif line.startswith('Threshold='):
                        self.threshold = float(line.strip().split('=')[1])
            print(f"Loaded params: T={self.T}, Threshold={self.threshold}")
        else:
            print("Warning: Calibration file not found. Using defaults.")

        self.labels = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess(self, frame):
        # Convert BGR (OpenCV) to RGB (PIL)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        tensor = self.transform(pil_image).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, frame):
        input_tensor = self.preprocess(frame)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            
            # Energy Calculation
            # E(x) = -T * logsumexp(logits/T)
            energy = -self.T * torch.logsumexp(logits / self.T, dim=1).item()
            
            # Confidence
            probs = F.softmax(logits / self.T, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
            
            status = "PASS"
            color = (0, 255, 0) # Green
            
            # Decision Logic
            # Note: ID has LOWER Energy (more negative). OOD has HIGHER Energy.
            # If Energy > Threshold -> OOD
            if energy > self.threshold:
                status = "BLOCK (OOD)"
                color = (0, 0, 255) # Red
            elif confidence.item() < 0.6:
                status = "LOW CONFIDENCE"
                color = (0, 255, 255) # Yellow
                
            return {
                "label": self.labels[pred_idx.item()],
                "confidence": confidence.item(),
                "energy": energy,
                "status": status,
                "color": color
            }

def main(args):
    detector = OODDetector(args.model_path, args.calibration_file)
    
    cap = cv2.VideoCapture(args.camera_id)
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Starting Inference... Press 'q' to quit.")
    
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Inference
        result = detector.predict(frame)
        
        # Visualization
        label_text = f"{result['label']} ({result['confidence']:.2f})"
        energy_text = f"Energy: {result['energy']:.2f} (Th: {detector.threshold:.2f})"
        status_text = f"Status: {result['status']}"
        
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, result['color'], 2)
        cv2.putText(frame, energy_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, result['color'], 2)
        
        # FPS Calculation
        fps_counter += 1
        if (time.time() - fps_start_time) > 1.0:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
            
        cv2.putText(frame, f"FPS: {fps:.1f}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        cv2.imshow('OOD Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--calibration_file', type=str, default='checkpoints/calibration_params.txt')
    parser.add_argument('--camera_id', type=int, default=0)
    parser.add_argument('--image_path', type=str, default=None, help='Path to image file for testing')
    args = parser.parse_args()
    
    if args.image_path:
        # Single Image Inference
        if not os.path.exists(args.image_path):
            print(f"Error: Image {args.image_path} not found.")
            sys.exit(1)
            
        frame = cv2.imread(args.image_path)
        if frame is None:
            print("Error: Could not read image.")
            sys.exit(1)

        # Initialize detector here
        detector = OODDetector(args.model_path, args.calibration_file)
            
        result = detector.predict(frame)
        print("Inference Result:")
        print(f"Label: {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Energy: {result['energy']:.4f}")
        print(f"Status: {result['status']}")
        
        # Save result image
        label_text = f"{result['label']} ({result['confidence']:.2f})"
        energy_text = f"Energy: {result['energy']:.2f} (Th: {detector.threshold:.2f})"
        status_text = f"Status: {result['status']}"
        
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, result['color'], 2)
        cv2.putText(frame, energy_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, result['color'], 2)
        
        output_path = "inference_result.jpg"
        cv2.imwrite(output_path, frame)
        print(f"Result saved to {output_path}")
        sys.exit(0)

    main(args)
