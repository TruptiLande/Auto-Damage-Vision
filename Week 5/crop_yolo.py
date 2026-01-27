# Confidence threshold (adjust if needed)
CONF_THRESHOLD = 0.4

# Iterate over all images in the input directory
for img_name in os.listdir(INPUT_DIR):
    img_path = os.path.join(INPUT_DIR, img_name)

    # Read image
    image = cv2.imread(img_path)
    if image is None:
        continue

    h, w, _ = image.shape

    # Run YOLOv8 inference
    results = model(image)

    # Each image has one result object
    result = results[0]

    # Iterate over detected bounding boxes
    for i, box in enumerate(result.boxes):
        conf = float(box.conf[0])

        # Skip low-confidence detections
        if conf < CONF_THRESHOLD:
            continue

        # Bounding box coordinates (xyxy format)
        x1, y1, x2, y2 = box.xyxy[0]

        # Convert to integers and clamp to image boundaries
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        # Crop the damage region
        crop = image[y1:y2, x1:x2]

        # Skip empty or very small crops
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            continue

        # Create a unique filename
        base_name = os.path.splitext(img_name)[0]
        crop_name = f"{base_name}_damage_{i}.jpg"
        crop_path = os.path.join(OUTPUT_DIR, crop_name)

        # Save cropped image
        cv2.imwrite(crop_path, crop)

print("✅ Cropping completed. Cropped damage regions saved.")
