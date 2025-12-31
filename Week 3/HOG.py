# ==============================
# YOLOv3 Inference on Vehicle Damage Dataset
# Complete Single-Cell Script
# ==============================

# ---- Install dependencies ----
import sys
import subprocess

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "opencv-python-headless", "matplotlib", "kagglehub"])

# ---- Imports ----
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import kagglehub

# ---- Step 1: Download Dataset using kagglehub ----
dataset_path = kagglehub.dataset_download(
    "hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection"
)
print("Dataset path:", dataset_path)

# ---- Step 2: Collect all images recursively ----
image_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            image_files.append(os.path.join(root, file))

print(f"Found {len(image_files)} images")

# Safety check
if len(image_files) == 0:
    raise RuntimeError("No images found in dataset. Dataset structure may be incorrect.")

# Load a few images for inference
images = []
for f in image_files[:5]:
    img = cv2.imread(f)
    if img is not None:
        images.append(img)

print(f"Loaded {len(images)} images")

# ---- Step 3: Clone YOLOv3 repo (for cfg) ----
if not os.path.exists("darknet"):
    subprocess.run(["git", "clone", "https://github.com/pjreddie/darknet"])

# ---- Step 4: Download YOLOv3 pretrained weights ----
if not os.path.exists("yolov3.weights"):
    subprocess.run([
        "wget",
        "https://pjreddie.com/media/files/yolov3.weights"
    ])

# ---- Step 5: Load YOLOv3 model ----
cfg_path = "darknet/cfg/yolov3.cfg"
weights_path = "yolov3.weights"

net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# ---- Get output layers (OpenCV-version safe) ----
layer_names = net.getLayerNames()
unconnected = net.getUnconnectedOutLayers()

if isinstance(unconnected, tuple) or len(unconnected.shape) == 1:
    output_layers = [layer_names[i - 1] for i in unconnected]
else:
    output_layers = [layer_names[i[0] - 1] for i in unconnected]

print("YOLO output layers:", output_layers)

# ---- Step 6: Run YOLOv3 inference ----
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

for img in images:
    height, width = img.shape[:2]

    blob = cv2.dnn.blobFromImage(
        img, 1 / 255.0, (416, 416), swapRB=True, crop=False
    )
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONF_THRESHOLD:
                cx = int(detection[0] * width)
                cy = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(cx - w / 2)
                y = int(cy - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # ---- Non-Max Suppression (fully safe handling) ----
    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD
    )

    if len(indices) > 0:
        if isinstance(indices, tuple):
            indices = indices[0]

        for i in np.array(indices).flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"Damage: {confidences[i]:.2f}"
            cv2.putText(
                img, label, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )

    # ---- Step 7: Display result ----
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

print("YOLOv3 inference completed successfully.")
