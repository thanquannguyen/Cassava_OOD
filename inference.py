import torch
import cv2
import torch.nn.functional as F
import numpy as np
import argparse
import time
import os
import sys
import random
import base64
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
        self.roi = None
        if mqtt_broker:
            self.mqtt = MQTTClient(broker=mqtt_broker)
            self.mqtt.subscribe("roi", self.handle_roi)
            self.mqtt.subscribe("control", self.handle_control)
            self.mqtt.subscribe("config", self.handle_config)
            
        self.state = "PAUSED" # PAUSED, RUNNING
        self.single_shot_path = None
        self.delay = 2.0 # Default delay
            
        # Load Model
        # Use pretrained=False to avoid downloading ImageNet weights every time. 
        # We load our own state_dict immediately after, so random init is fine.
        print("Initializing model...", flush=True)
        self.model = CassavaMobileNet(num_classes=5, pretrained=False).to(self.device)
        
        # Check for Robust Model first
        robust_path = model_path.replace(".pth", "_robust.pth")
        if os.path.exists(robust_path):
            print(f"--> Found ROBUST model: {robust_path}. Loading...", flush=True)
            self.model.load_state_dict(torch.load(robust_path, map_location=self.device))
        else:
            print(f"--> Loading standard model: {model_path}", flush=True)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            
        self.model.eval()
        print("Model loaded successfully.", flush=True)
        
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
        
        # Improved Preprocessing: Resize distinct to (224, 224) to see the WHOLE image
        # Prevents missing disease symptoms at the edges (User Feedback)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("  [Debug] Transforms initialized.", flush=True)

    def handle_roi(self, payload):
        # Payload is {x, y, w, h} normalized 0-1 or Empty {} for reset
        if payload and 'x' in payload:
            self.roi = payload
            print(f"[MQTT] ROI Updated: {self.roi}")
        else:
            self.roi = None
            print(f"[MQTT] ROI Reset")

    def preprocess(self, frame):
        # Convert BGR (OpenCV) to RGB (PIL)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        tensor = self.transform(pil_image).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, frame, custom_roi=None):
        # Handle ROI Cropping
        if custom_roi:
            h, w = frame.shape[:2]
            rx, ry, rw, rh = custom_roi['x'], custom_roi['y'], custom_roi['w'], custom_roi['h']
            
            # Convert normalized to pixel coords
            x1 = int(rx * w)
            y1 = int(ry * h)
            x2 = int((rx + rw) * w)
            y2 = int((ry + rh) * h)
            
            # Clamp
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            
            if x2 > x1 and y2 > y1:
                frame_analytics = frame[y1:y2, x1:x2] # Crop for analysis
                # print(f"  [Debug] Cropped to {x1}:{x2}, {y1}:{y2}")
            else:
                 frame_analytics = frame # Fallback
        else:
             frame_analytics = frame

        input_tensor = self.preprocess(frame_analytics)
        
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
                "color": color,
                "path": "" # Will be filled by main
            }

    def publish_mqtt(self, result, frame):
        if self.mqtt:
             # Encode Frame to Base64 (Annotated Frame)
            _, buffer = cv2.imencode('.jpg', frame)
            b64_str = base64.b64encode(buffer).decode('utf-8')
            
            # Add to result
            result['image'] = b64_str
            
            self.mqtt.publish("detection", result)
            if result['status'] == "BLOCK (OOD)":
                self.mqtt.publish("alert", {"message": "OOD Detected!", "energy": result['energy']})

    def draw_ui(self, frame, result, fps=None):
        h, w = frame.shape[:2]
        
    def draw_ui(self, frame, result, fps=None):
        h, w = frame.shape[:2]
        
        # 1. Draw Focus Box
        if self.roi:
            # User defined ROI (Green Box)
            rx, ry, rw, rh = self.roi['x'], self.roi['y'], self.roi['w'], self.roi['h']
            x1 = int(rx * w)
            y1 = int(ry * h)
            x2 = int((rx + rw) * w)
            y2 = int((ry + rh) * h)
             # Clamp
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), result['color'], 3)
            # Label near box
            cv2.putText(frame, "Interactive ROI", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, result['color'], 1)

        else:
            # Default Center Box (for Live Cam or Reset)
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

    def handle_control(self, payload):
        # {cmd: "start"|"stop"|"load_image", path: "..."}
        cmd = payload.get('cmd')
        if cmd == 'start':
            self.state = "RUNNING"
            print("[Control] System STARTED")
        elif cmd == 'stop':
            self.state = "PAUSED"
            print("[Control] System STOPPED")
        elif cmd == 'load_image':
            filename = payload.get('path')
            # Resolve full path from image_map (if available)
            full_path = None
            if hasattr(self, 'image_map') and self.image_map:
                full_path = self.image_map.get(filename)
            
            # Fallback: Check if path is already valid
            if not full_path and os.path.exists(filename):
                full_path = filename
                
            if full_path and os.path.exists(full_path):
                self.single_shot_path = full_path
                print(f"[Control] Loading image: {full_path}")
            else:
                print(f"[Control] Error: Image not found for {filename}")

    def handle_config(self, payload):
        # {threshold: float, delay: float}
        if 'threshold' in payload:
            self.threshold = float(payload['threshold'])
            print(f"[Config] Threshold updated to {self.threshold}")
        if 'delay' in payload:
            self.delay = float(payload['delay'])
            print(f"[Config] Delay updated to {self.delay}")

    # Disease Acronym Mapping - MOVED TO MAIN FOR SCOPE SAFETY
    # full_names defined in main now

def main(args):
    # Disease Acronym Mapping
    full_names = {
        "CBB": "Cassava Bacterial Blight",
        "CBSD": "Cassava Brown Streak Disease",
        "CGM": "Cassava Green Mottle",
        "CMD": "Cassava Mosaic Disease",
        "Healthy": "Healthy"
    }

    detector = OODDetector(args.model_path, args.calibration_file, mqtt_broker=args.mqtt_broker)
    
    # Apply Overrides if provided
    if args.override_temperature is not None:
        detector.T = args.override_temperature
        print(f"Override applied: T={detector.T}")
    if args.override_threshold is not None:
        detector.threshold = args.override_threshold
        print(f"Override applied: Threshold={detector.threshold}")
    
    # Init delay from args
    detector.delay = args.delay
    
    cap = None
    frame_source = None
    image_files = []
    
    if args.save_output:
        os.makedirs(args.output_dir, exist_ok=True)
    
    if args.image_path:
        if os.path.isdir(args.image_path):
            # Directory Mode (Recursive)
            valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
            image_files = []
            print(f"Scanning {args.image_path} recursively...", flush=True)
            for root, dirs, files in os.walk(args.image_path):
                for f in files:
                    if f.lower().endswith(valid_exts):
                        image_files.append(os.path.join(root, f))
                        
            if not image_files:
                print(f"Error: No images found in {args.image_path} or subdirectories")
                sys.exit(1)
            print(f"Mode: Directory Simulation ({len(image_files)} images found)")
            
            # Populate Image Map for Control System
            detector.image_map = {os.path.basename(p): p for p in image_files}
            print(f"  [Control] Indexed {len(detector.image_map)} images for review.")
            
            args.loop = True # Force loop mode for directory
            
        elif os.path.exists(args.image_path):
            # Single Image Mode
            print(f"Reading image {args.image_path}...", flush=True)
            try:
                # Use PIL to verify readability
                Image.open(args.image_path).verify()
                # Re-open for actual usage (verify closes the file)
                # We defer loading to the loop for directory, but load here for single
                pil_img = Image.open(args.image_path).convert('RGB')
                frame_source = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"  [Error] Failed to read image with PIL: {e}", flush=True)
                sys.exit(1)
            print(f"Loaded image: {args.image_path}")
            if args.loop:
                print("Mode: Loop (Simulation)")
            else:
                print("Mode: Single Shot")
        else:
            print(f"Error: Path {args.image_path} not found.")
            sys.exit(1)
            
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
    
            
    
    detector.state = "PAUSED" # Start in PAUSED mode as requested
    print("System Output: PAUSED. Waiting for Start command...", flush=True)

    while True:
        # Check for Single Shot Request (from Table Click)
        if detector.single_shot_path:
             try:
                 pil_img = Image.open(detector.single_shot_path).convert('RGB')
                 frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                 current_filename = os.path.basename(detector.single_shot_path)
                 print(f"Processing SINGLE SHOT: {current_filename}")
                 # Clear request
                 detector.single_shot_path = None
             except Exception as e:
                 print(f"Error reading {detector.single_shot_path}: {e}")
                 detector.single_shot_path = None
                 continue
        
        elif detector.state == "PAUSED":
            time.sleep(0.5)
            continue
            
        elif cap:
        # ... (Camera Logic remains same, runs if state=RUNNING)
            ret, frame = cap.read()
            if not ret: break
            current_filename = "Live Camera"
        else:
            # Simulation Source (Running)
            if image_files:
                img_path = random.choice(image_files)
                try:
                    pil_img = Image.open(img_path).convert('RGB')
                    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    current_filename = os.path.basename(img_path)
                except Exception as e:
                    print(f"Error reading {img_path}: {e}")
                    continue
            else:
                 frame = frame_source.copy()
                 current_filename = os.path.basename(args.image_path) if args.image_path else "image"

        # Inference
        # If Camera Mode: Ignore ROI (User Request: "live cam... bounding box trung tâm")
        active_roi = None
        if not cap and detector.roi: 
             active_roi = detector.roi

        result = detector.predict(frame, custom_roi=active_roi)
        # Add filename/path
        result['filename'] = current_filename
        result['path'] = current_filename

        # Visualization FIRST (to annotate frame before sending)
        if not args.headless or args.save_output or True: # Force draw for Web UI
            detector.draw_ui(frame, result, fps=fps)
        
        # Publish to MQTT (includes Annotated Image)
        detector.publish_mqtt(result, frame) # Sending the annotated frame now!
        
        if args.headless:
            # In headless directory simulation, print every result but with details
            full_name = full_names.get(result['label'], result['label'])
            print(f"Processed: {result['label']} ({full_name}) | Energy: {result['energy']:.2f}", flush=True)

        # Save Output
        if args.save_output:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            # If using directory mode, try to keep original filename prefix
            prefix = "sim"
            if image_files and 'img_path' in locals():
                prefix = os.path.splitext(os.path.basename(img_path))[0]
            elif args.image_path and os.path.isfile(args.image_path):
                 prefix = os.path.splitext(os.path.basename(args.image_path))[0]
                 
            filename = f"{prefix}_{timestamp}.jpg"
            save_path = os.path.join(args.output_dir, filename)
            cv2.imwrite(save_path, frame)
            # print(f"Saved {save_path}", flush=True)

        # Simulation Delay (Using detector.delay which can be updated live)
        if (args.loop or args.image_path) and detector.state == "RUNNING":
            time.sleep(detector.delay)
        
        # FPS Calculation
        
        # FPS Calculation
        fps_counter += 1
        if (time.time() - fps_start_time) > 1.0:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
            
            fps_start_time = time.time()
        
        if not args.headless:
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
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no GUI window)')
    parser.add_argument('--delay', type=float, default=2.0, help='Delay in seconds between frames in simulation mode')
    parser.add_argument('--save_output', action='store_true', help='Save processed images to disk')
    parser.add_argument('--output_dir', type=str, default='output_results', help='Directory to save output images')
    parser.add_argument('--override_threshold', type=float, default=None, help='Manually override OOD threshold')
    parser.add_argument('--override_temperature', type=float, default=None, help='Manually override Temperature T')
    args = parser.parse_args()
    
    main(args)
