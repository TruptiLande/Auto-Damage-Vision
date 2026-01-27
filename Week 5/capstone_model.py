# Week 5 – End-to-End Vehicle Damage Severity Pipeline
# Objective: Integrate YOLOv8 detection + severity classification

# Step 1: Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import tensorflow as tf
import os

# Step 2: Load Models
# Load YOLOv8 damage detector (Week 4)
yolo_model = YOLO("best.pt")

# Load trained severity classifier (Week 5)
severity_model = tf.keras.models.load_model("severity_classifier.h5")

# Step 3: Load Image
IMAGE_PATH = "test_vehicle.jpg"
image = cv2.imread(IMAGE_PATH)

assert image is not None, "Image not found!"

# Step 4: Detect Damage Regions
CONF_THRESHOLD = 0.4
boxes = []

results = yolo_model(image)
result = results[0]

for box in result.boxes:
    conf = float(box.conf[0])
    if conf < CONF_THRESHOLD:
        continue

    x1, y1, x2, y2 = box.xyxy[0]
    boxes.append([int(x1), int(y1), int(x2), int(y2)])

# Step 5: Predict Severity
severity_labels = ["Minor", "Moderate", "Severe"]
IMG_SIZE = 128  # must match training size

h, w, _ = image.shape

for box in boxes:
    x1, y1, x2, y2 = box

    # Clamp box to image boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Crop damage region
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        continue

    # Preprocess for CNN
    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    crop = crop / 255.0
    crop = np.expand_dims(crop, axis=0)

    # Predict severity
    preds = severity_model.predict(crop, verbose=0)
    severity_idx = np.argmax(preds)
    severity = severity_labels[severity_idx]

    # Draw bounding box and label
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        image,
        severity,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

# Step 6: Visualization
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# Reflection:
# - If YOLO detection is incorrect, severity prediction either fails or becomes unreliable
# - Errors propagate downstream from detection to classification
# - Modular design allows independent improvement of detector and classifier
# - Severity is a semantic concept and requires context beyond bounding box size
