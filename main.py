import cv2
import numpy as np
import onnxruntime as ort
import paho.mqtt.client as mqtt
import os
import sys
import json
import time
from flask import Flask, jsonify

# --- ENVIRONMENT & CONFIGURATION ---
RTSP_URL = os.getenv("RTSP_URL", "rtsp://username:password@YOUR_CAMERA_IP:554/stream1")

MQTT_BROKER = os.getenv("MQTT_BROKER", "YOUR_MQTT_BROKER_IP")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_USER = os.getenv("MQTT_USER", None)
MQTT_PASS = os.getenv("MQTT_PASS", None)

CROPS = {
    "garage_1": (443, 877, 621, 1086),      # Left Garage Door
    "garage_2": (460, 874, 1192, 1650),    # Right Garage Door
    "car_1":    (908, 1276, 568, 1113),   # Parking Spot 1
    "car_2":    (929, 1296, 1681, 2304)   # Parking Spot 2
}

ENTITIES = {
    "garage_1": {
        "model_path": "models/garage_1_model.onnx",
        "labels": ["closed", "open"],
        "topic": "homeassistant/sensor/garage_door_1/state"
    },
    "garage_2": {
        "model_path": "models/garage_2_model.onnx",
        "labels": ["closed", "open"],
        "topic": "homeassistant/sensor/garage_door_2/state"
    },
    "car_1": {
        "model_path": "models/car_1_model.onnx",
        "labels": ["absent", "present"],
        "topic": "homeassistant/sensor/parking_spot_1/state"
    },
    "car_2": {
        "model_path": "models/car_2_model.onnx",
        "labels": ["absent", "present"],
        "topic": "homeassistant/sensor/parking_spot_2/state"
    }
}

# --- INITIALIZE ONNX SESSIONS ---
print("[*] Loading ONNX Models...")
sessions = {}
for name, config in ENTITIES.items():
    model_path = config["model_path"]
    if not os.path.exists(model_path):
        print(f"[!] Error: Model file not found at '{model_path}'.")
        sys.exit(1)
    
    sessions[name] = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    print(f"    Loaded model: {name}")

# --- MQTT SETUP ---
mqtt_client = mqtt.Client(
    callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    protocol=mqtt.MQTTv5
)

if MQTT_USER and MQTT_PASS:
    mqtt_client.username_pw_set(MQTT_USER, MQTT_PASS)

def publish_discovery():
    """Publishes Home Assistant MQTT Auto-Discovery configuration payloads."""
    for name, config in ENTITIES.items():
        is_garage = "garage" in name
        
        unique_id = f"garage_vision_{name}"
        discovery_topic = f"homeassistant/binary_sensor/garage_vision_{name}/config"
        
        payload = {
            "name": f"Garage Vision {name.replace('_', ' ').title()}",
            "unique_id": unique_id,
            "state_topic": config["topic"],
            "payload_on": "open" if is_garage else "present",
            "payload_off": "closed" if is_garage else "absent",
            "device_class": "garage_door" if is_garage else "occupancy",
            "device": {
                "identifiers": ["garage_vision_ai_system"],
                "name": "Garage Vision AI System",
                "model": "MobileNetV2 ONNX",
                "manufacturer": "Custom AI"
            }
        }
        
        # Publish discovery config as retained message
        mqtt_client.publish(discovery_topic, json.dumps(payload), retain=True)
        print(f"[*] Published HA Auto-Discovery config for: {name}")

def connect_mqtt():
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        print(f"[*] Connected to MQTT Broker at {MQTT_BROKER}:{MQTT_PORT}")
        # Pause briefly to ensure the MQTT network thread establishes the socket handshake
        time.sleep(1)
        publish_discovery()
    except Exception as e:
        print(f"[!] MQTT Connection error: {e}")

connect_mqtt()

# --- PREPROCESSING & INFERENCE ---
def preprocess(crop_img):
    img = cv2.resize(crop_img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def process_frame():
    cap = cv2.VideoCapture(RTSP_URL)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("[!] Failed to pull RTSP frame.")
        return {"status": "error", "message": "Failed to read RTSP frame"}

    results = {}
    for name, (y1, y2, x1, x2) in CROPS.items():
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        input_data = preprocess(cropped)
        session = sessions[name]
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_data})
        
        class_idx = int(np.argmax(outputs[0][0]))
        state_label = ENTITIES[name]["labels"][class_idx]
        topic = ENTITIES[name]["topic"]
        
        mqtt_client.publish(topic, state_label, retain=True)
        results[name] = state_label
        print(f"[TRIGGERED INFERENCE] {name} -> {state_label}")

    return {"status": "success", "results": results}

# --- FLASK SERVER ---
app = Flask(__name__)

@app.route('/trigger', methods=['POST', 'GET'])
def trigger_inference():
    res = process_frame()
    return jsonify(res)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "models_loaded": len(sessions)})

if __name__ == "__main__":
    print("[*] Starting Flask trigger server on port 5000...")
    app.run(host="0.0.0.0", port=5000)
