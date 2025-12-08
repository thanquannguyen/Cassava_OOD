import paho.mqtt.client as mqtt
import json
import time

BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "cassava/#"

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe(TOPIC)
    print(f"Subscribed to {TOPIC}")

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        data = json.loads(payload)
        print(f"\n[{msg.topic}] Received:")
        print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"\n[{msg.topic}] Raw: {msg.payload}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

print(f"Connecting to {BROKER}...")
client.connect(BROKER, PORT, 60)

try:
    client.loop_forever()
except KeyboardInterrupt:
    print("\nExiting...")
    client.disconnect()
