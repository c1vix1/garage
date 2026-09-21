import os
import time
import cv2
import numpy as np
import paho.mqtt.client as mqtt
from openvino.runtime import Core

# --- Configuration from Environment Variables ---
# Stream for Car 1 physical camera (now running the model for Car 2)
RTSP_URL = os.getenv("RTSP_URL", "rtsp://username:password@192.168.1.2:554/stream1")

MQTT_BROKER = os.getenv("MQTT_BROKER", "192.168.1.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_USER = os.getenv("MQTT_USER", None)
MQTT_PASS = os.getenv("MQTT_PASS", None)

CROPS = {
    "garage_1": (443, 877, 621, 1086),      # Left Garage Door
    "garage_2": (460, 874, 1192, 1650),    # Right Garage Door
    "car_1":    (908, 1276, 568, 1113),   # Parking Spot 1
    "car_2":    (929, 1296, 1681, 2304)   # Parking Spot 2
}

# --- MQTT Setup ---
client = mqtt.Client()
if MQTT_USER and MQTT_PASS:
    client.username_pw_set(MQTT_USER, MQTT_PASS)

client.connect(MQTT_BROKER, MQTT_PORT, 60)

# 1. Clear old retained discovery topic for Garage Door 1 (so HA removes it)
client.publish("homeassistant/binary_sensor/garage_door_1/config", "", retain=True)

# 2. Register Home Assistant Discovery for Garage Door 2 (Swap: Car 1 video -> Car 2 entity)
discovery_topic_2 = "homeassistant/binary_sensor/garage_door_2/config"
discovery_payload_2 = '''{
  "name": "Garage Door 2 State",
  "state_topic": "garage/door_2/state",
  "device_class": "garage_door",
  "unique_id": "garage_door_vision_model_2"
}'''
client.publish(discovery_topic_2, discovery_payload_2, retain=True)

# --- OpenVINO Vision Setup (Intel HD 630 iGPU) ---
core = Core()
# Model path (adjust filename if using ONNX or OpenVINO XML)
model = core.read_model(model="models/garage_door_model.xml")
compiled_model = core.compile_model(model=model, device_name="GPU")
output_layer = compiled_model.output(0)


def process_and_publish():
    cap = cv2.VideoCapture(RTSP_URL)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("Failed to capture frame from RTSP stream.")
        return

    # Preprocessing image (224x224 input)
    img = cv2.resize(frame, (224, 224)).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    input_tensor = np.expand_dims(img, axis=0)

    # Run inference on Intel iGPU
    results = compiled_model([input_tensor])[output_layer]
    prediction = np.argmax(results)

    # State calculation (1 = Open, 0 = Closed)
    is_open = prediction == 1
    state_str = "ON" if is_open else "OFF"

    # Publish status specifically to the swapped "Garage Door 2" topic
    client.publish("garage/door_2/state", state_str, retain=True)
    print(f"Car 1 Stream -> Updated Garage Door 2 State: {'OPEN' if is_open else 'CLOSED'}")


# --- Main Loop ---
while True:
    try:
        process_and_publish()
    except Exception as e:
        print(f"Error running inference: {e}")
    time.sleep(CHECK_INTERVAL)