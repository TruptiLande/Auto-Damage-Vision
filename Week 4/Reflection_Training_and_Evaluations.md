1. What worked well?
Transfer Learning Efficiency: Starting with pretrained yolov8n.pt weights allowed the model to skip learning basic features (like lines, edges, and colors) and move straight to specialized features. This significantly reduced training time even on a CPU.
Metric Visibility: The YOLOv8 training engine provided clear, real-time feedback. Being able to see the box_loss and cls_loss decrease over epochs confirmed that the "math" of the model was successfully optimizing.
Framework Integration: Using a consolidated tool like ultralytics simplified the pipeline from dataset configuration (data.yaml) to the final evaluation of Precision and Recall.

2. What failed? (Error Analysis)
Domain Gap (Pre-training Phase): Initially, the model had a near-zero mAP because it was trained for general objects (people/bikes). This gap proved that general AI is ineffective for specialized insurance or automotive tasks without custom training.
Pathing and Environment Instability: The FileNotFoundError and ModuleNotFoundError highlighted the sensitivity of cloud environments like Colab. Small issues with relative vs. absolute paths in the YAML file can prevent an otherwise perfect model from training.
Small Target Misses: During visual evaluation, the model likely struggled with very fine scratches (tray_son) compared to large dents. Small batch sizes (batch=8) on a CPU also made the training "noisier," meaning the metrics might have fluctuated rather than improving smoothly.

3. What would you improve with more data or time?
Slicing-Aided Hyper Inference (SAHI): For high-resolution images where a scratch is only a few pixels wide, I would implement SAHI. This technique "slices" the image into smaller patches, allowing the model to see tiny damages more clearly.
Data Augmentation: I would add techniques like Mosaic augmentation, random brightness adjustments (to simulate night/day inspections), and flipping. This would help the model generalize better despite the small subset of 160 images.
Class Balancing: If the model continues to miss specific classes like rach (cracks), I would gather more examples of that specific damage or use "Oversampling" to show the model those rare instances more frequently during training.
Hyperparameter Evolution: With more time, I would run a "Hyperparameter Tuning" session to find the perfect learning rate (lr0) and momentum, which could push the mAP score from "decent" to "production-ready."
