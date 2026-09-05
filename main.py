import cv2
import numpy as np
import onnxruntime as ort
import paho.mqtt.client as mqtt
import os
import sys
from flask import Flask, jsonify

# --- ENVIRONMENT & CONFIGURATION ---
# Default placeholders ensure safe GitHub uploads without leaking local IPs or credentials
RTSP_URL = os.getenv("RTSP_URL", "rtsp://username:password@YOUR_CAMERA_IP:554/stream1")

MQTT_BROKER = os.getenv("MQTT_BROKER", "YOUR_MQTT_BROKER_IP")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_USER = os.getenv("MQTT_USER", None)
MQTT_PASS = os.getenv("MQTT_PASS", None)

# Region of Interest (ROI) crop coordinates: (y1, y2, x1, x2)
CROPS = {
    "garage_1": (0, 720, 0, 450),
    "garage_2": (0, 720, 450, 900),
    "car_1":    (150, 720, 100, 600),
    "car_2":    (150, 720, 600, 1100)
}

# Entities setup: model file location, binary class names, and MQTT state topic
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
print("[*] Loading ONNX Models into CPU Execution Provider...")
sessions = {}
for name, config in ENTITIES.items():
    model_path = config["model_path"]
    if not os.path.exists(model_path):
        print(f"[!] Error: Model file not found at '{model_path}'. Check your Docker mount/copy paths.")
        sys.exit(1)
    
    sessions[name] = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    print(f"    Loaded model: {name} ({model_path})")

# --- INITIALIZE MQTT CLIENT ---
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
if MQTT_USER and MQTT_PASS:
    mqtt_client.username_pw_set(MQTT_USER, MQTT_PASS)

def connect_mqtt():
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        print(f"[*] Successfully connected to MQTT Broker at {MQTT_BROKER}:{MQTT_PORT}")
    except Exception as e:
        print(f"[!] MQTT Connection error: {e}")

connect_mqtt()

# --- PREPROCESSING & INFERENCE ---
def preprocess(crop_img):
    """Resizes and normalizes the cropped region to match PyTorch MobileNetV2 defaults."""
    img = cv2.resize(crop_img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    
    # Standard ImageNet Mean and Standard Deviation
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    
    # Transpose from (H, W, C) to (C, H, W) and expand to batch shape (1, C, H, W)
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def process_frame():
    """Captures a snapshot from RTSP, runs all four models, and publishes results to MQTT."""
    cap = cv2.VideoCapture(RTSP_URL)
    ret, frame = cap.read()
    cap.release()  # Release camera connection immediately to avoid lingering RTSP streams

    if not ret or frame is None:
        print("[!] Failed to pull RTSP frame from camera.")
        return {"status": "error", "message": "Failed to read RTSP frame from camera"}

    results = {}
    for name, (y1, y2, x1, x2) in CROPS.items():
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0 or cropped.shape[0] == 0 or cropped.shape[1] == 0:
            print(f"[Warning] Invalid crop region for entity '{name}'. Skipping.")
            continue

        # Prepare image tensor
        input_data = preprocess(cropped)
        
        # Execute ONNX Inference
        session = sessions[name]
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_data})
        
        # Get highest probability index and map to label name
        class_idx = int(np.argmax(outputs[0][0]))
        state_label = ENTITIES[name]["labels"][class_idx]
        topic = ENTITIES[name]["topic"]
        
        # Publish retained state message to MQTT
        mqtt_client.publish(topic, state_label, retain=True)
        results[name] = state_label
        print(f"[TRIGGERED INFERENCE] {name} -> {state_label} (Published to '{topic}')")

    return {"status": "success", "results": results}

# --- FLASK REST API SERVER ---
app = Flask(__name__)

@app.route('/trigger', methods=['POST', 'GET'])
def trigger_inference():
    """Endpoint called by Home Assistant REST command when triggered by an automation."""
    res = process_frame()
    return jsonify(res)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Docker / Portainer monitoring."""
    return jsonify({"status": "healthy", "models_loaded": len(sessions)})

if __name__ == "__main__":
    print("[*] Starting Flask trigger server on port 5000...")
    app.run(host="0.0.0.0", port=5000)
