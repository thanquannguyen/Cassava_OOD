import paho.mqtt.client as mqtt
import json
import time
import threading

class MQTTClient:
    def __init__(self, broker="broker.hivemq.com", port=1883, topic_prefix="cassava"):
        self.broker = broker
        self.port = port
        self.topic_prefix = topic_prefix
        self.client = mqtt.Client()
        self.connected = False
        
        # Callbacks
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        
        self.message_handlers = {}
        
        # Start background loop
        self.start()

    def start(self):
        try:
            print(f"[MQTT] Connecting to {self.broker}:{self.port}...")
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
        except Exception as e:
            print(f"[MQTT] Connection failed: {e}")

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"[MQTT] Connected to {self.broker}")
            self.connected = True
            # Resubscribe to topics
            for subtopic in self.message_handlers:
                topic = f"{self.topic_prefix}/{subtopic}"
                client.subscribe(topic)
                print(f"[MQTT] Subscribed to {topic}")
        else:
            print(f"[MQTT] Failed to connect, return code {rc}")
            self.connected = False

    def on_message(self, client, userdata, msg):
        try:
            # Extract subtopic (remove prefix)
            topic = msg.topic
            if topic.startswith(f"{self.topic_prefix}/"):
                subtopic = topic[len(self.topic_prefix)+1:]
                if subtopic in self.message_handlers:
                    payload = json.loads(msg.payload.decode())
                    self.message_handlers[subtopic](payload)
        except Exception as e:
            print(f"[MQTT] Error handling message: {e}")

    def subscribe(self, subtopic, callback):
        self.message_handlers[subtopic] = callback
        if self.connected:
            topic = f"{self.topic_prefix}/{subtopic}"
            self.client.subscribe(topic)
            print(f"[MQTT] Subscribed to {topic}")

    def on_disconnect(self, client, userdata, rc):
        print(f"[MQTT] Disconnected (rc={rc})")
        self.connected = False

    def publish(self, subtopic, data):
        if not self.connected:
            return
            
        topic = f"{self.topic_prefix}/{subtopic}"
        try:
            payload = json.dumps(data)
            self.client.publish(topic, payload)
            # print(f"[MQTT] Published to {topic}: {payload}")
        except Exception as e:
            print(f"[MQTT] Publish failed: {e}")

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()
