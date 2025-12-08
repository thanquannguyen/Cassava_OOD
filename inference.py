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
from utils.mqtt_client import MQTTClient

# Check if TensorRT is available (optional optimization)
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

class OODDetector:
    def __init__(self, model_path, calibration_file, device='cuda', mqtt_broker=None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # MQTT Setup
        self.mqtt = None
        if mqtt_broker:
            self.mqtt = MQTTClient(broker=mqtt_broker)
        
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

    def publish_mqtt(self, result):
        if self.mqtt:
            self.mqtt.publish("detection", result)
            if result['status'] == "BLOCK (OOD)":
                self.mqtt.publish("alert", {"message": "OOD Detected!", "energy": result['energy']})

    def draw_ui(self, frame, result, fps=None):
        h, w = frame.shape[:2]
        
        # 1. Draw Focus Box (Center)
        box_size = 224
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), result['color'], 2)
        
        # 2. Prepare Text Info
        lines = [
            f"Label: {result['label']}",
            f"Conf: {result['confidence']:.2f}",
            f"Energy: {result['energy']:.2f}",
            f"Status: {result['status']}"
        ]
        if fps:
            lines.append(f"FPS: {fps:.1f}")
            
        # 3. Draw Text with Background
        start_y = 30
        for line in lines:
            (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (5, start_y - text_h - 5), (5 + text_w + 10, start_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, line, (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, result['color'], 2)
            start_y += 35
            
        return frame

def main(args):
    detector = OODDetector(args.model_path, args.calibration_file, mqtt_broker=args.mqtt_broker)
    
    cap = None
    frame_source = None
    
    if args.image_path:
        if not os.path.exists(args.image_path):
            print(f"Error: Image {args.image_path} not found.")
            sys.exit(1)
        frame_source = cv2.imread(args.image_path)
        if frame_source is None:
            print("Error: Could not read image.")
            sys.exit(1)
        print(f"Loaded image: {args.image_path}")
        if args.loop:
            print("Mode: Loop (Simulation)")
        else:
            print("Mode: Single Shot")
            
    else:
        cap = cv2.VideoCapture(args.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        print("Mode: Camera Live")


    print("Starting Inference... Press 'q' to quit.")
    
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    while True:
        if cap:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            # Single Image Source
            frame = frame_source.copy() # Use copy to avoid drawing over original if looping logic changes
            if not args.loop:
                 # If not looping, we run once effectively
                 pass

            
        # Inference
        result = detector.predict(frame)
        
        # Publish to MQTT
        detector.publish_mqtt(result)
        
        # Visualization
        detector.draw_ui(frame, result, fps=fps)
        
        # FPS Calculation
        
        # FPS Calculation
        fps_counter += 1
        if (time.time() - fps_start_time) > 1.0:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
            
            fps_start_time = time.time()
        
        cv2.imshow('OOD Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        if not cap and not args.loop:
            # Single shot finished
            print(f"Result saved to inference_result.jpg (Single Shot)")
            cv2.imwrite("inference_result.jpg", frame)
            # Short pause to let user see it if they ran from GUI, but mainly we exit
            time.sleep(1) 
            break
            
    if cap:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--calibration_file', type=str, default='checkpoints/calibration_params.txt')
    parser.add_argument('--camera_id', type=int, default=0)
    parser.add_argument('--image_path', type=str, default=None, help='Path to image file for testing')
    parser.add_argument('--mqtt_broker', type=str, default=None, help='MQTT Broker address (e.g., broker.hivemq.com)')
    parser.add_argument('--loop', action='store_true', help='Loop the image to simulate a video stream')
    args = parser.parse_args()
    
    main(args)
