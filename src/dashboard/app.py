import eventlet
# Monkey patch for eventlet (required for SocketIO async)
eventlet.monkey_patch()

from flask import Flask, render_template
from flask_socketio import SocketIO
import paho.mqtt.client as mqtt
import json
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# MQTT Configuration
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "cassava/#"
mqtt_client = None  # Global client instance

# Global variables to store latest state
current_state = {
    "label": "Waiting...",
    "confidence": 0.0,
    "energy": 0.0,
    "status": "Waiting",
    "color": [128, 128, 128]
}

def on_connect(client, userdata, flags, rc):
    print(f"[Dashboard] Connected to MQTT Broker (rc={rc})")
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    global current_state
    try:
        topic = msg.topic
        payload = msg.payload.decode()
        data = json.loads(payload)
        
        if "detection" in topic:
            current_state = data
            # Emit to websocket clients
            socketio.emit('update_data', data)
            
        elif "alert" in topic:
            socketio.emit('alert', data)
            
    except Exception as e:
        print(f"[Dashboard] Error parsing message: {e}")

# Start MQTT Client in a background thread
def start_mqtt():
    global mqtt_client
    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    
    print(f"[Dashboard] Connecting to {MQTT_BROKER}...")
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"[Dashboard] MQTT Connection Failed: {e}")

@app.route('/')
def index():
    return render_template('index.html', initial_state=current_state)

@socketio.on('update_roi')
def handle_roi_update(roi_data):
    if mqtt_client:
        payload = json.dumps(roi_data) if roi_data else "{}"
        mqtt_client.publish("cassava/roi", payload)

@socketio.on('config')
def handle_config(data):
    # data: {threshold: float, delay: float}
    print(f"[Dashboard] Config Update: {data}")
    if mqtt_client:
        mqtt_client.publish("cassava/config", json.dumps(data))

@socketio.on('control')
def handle_control(data):
    # data: {cmd: "start"|"stop"|"load_image", path: "..."}
    print(f"[Dashboard] Control Command: {data}")
    if mqtt_client:
        mqtt_client.publish("cassava/control", json.dumps(data))

if __name__ == '__main__':
    start_mqtt()
    print("[Dashboard] Starting Web Server on port 5000...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)
