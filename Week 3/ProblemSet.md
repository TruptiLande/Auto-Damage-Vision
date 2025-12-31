Problem 1. What is an image encoding? Why are raw pixel values generally a poor encoding for visual recognition tasks?
Ans. Image encoding is a way of representing an image as a set of numbers (a feature vector or structured representation) that captures the important visual information in a form that a computer can process efficiently. The goal is to transform raw image data into a representation that makes tasks like classification, detection, or recognition easier. Raw pixel values are a poor encoding because they change drastically with small variations in lighting, noise, or viewpoint, lack semantic meaning about shapes or objects, have very high dimensionality, are not invariant to common transformations, and fail to capture meaningful spatial structures needed for reliable visual recognition.

Problem 2. Histogram of Oriented Gradients (HOG) captures edge information. Why is edge information useful for recognizing objects in images?
Ans. Edge information is useful for recognizing objects because edges encode the structure and shape of objects, which are often more stable and distinctive than raw pixel values.
Here are the key reasons:
1. Object shape is defined by edges
Edges correspond to boundaries between regions (e.g., object vs. background or different object parts). These boundaries outline an object’s shape, which is a strong cue for distinguishing one object from another (e.g., a car vs. a pedestrian).
2. Invariant to illumination and color changes
Edges depend mainly on intensity gradients rather than absolute pixel values. This makes them more robust to changes in lighting, shadows, and color variations, which commonly affect raw pixel-based features.
3. Captures local spatial structure
HOG summarizes the distribution of edge orientations in local regions. This preserves information about how edges are arranged (e.g., vertical, horizontal, diagonal), which helps identify characteristic patterns like limbs, corners, or contours.
4. Reduces sensitivity to noise and texture
By focusing on strong gradients and aggregating them over regions, edge-based features ignore small pixel-level noise and irrelevant texture, emphasizing meaningful object structure.
5. Shared across object instances
Different instances of the same object class often have similar edge configurations even if their appearance varies. For example, human silhouettes share consistent edge patterns regardless of clothing or color.

Problem 3. Early convolutional layers in CNNs often learn edge-like filters. Explain why this happens and how it relates to hand-crafted features such as HOG.
Ans. Early CNN layers learn edge-like filters because edges are the simplest and most informative visual patterns in natural images. With small convolutional kernels, the easiest way to reduce training loss is to detect local intensity changes, which correspond to edges. These features are common across images and useful for many tasks, so learning them early is efficient and reusable. This closely relates to hand-crafted features like HOG, which explicitly compute local gradient orientations to capture edge structure. Early CNN layers effectively learn HOG-like gradient detectors, but instead of using fixed, hand-designed filters, CNNs learn optimal edge orientations and combinations from data, and later layers build more complex features from them.

Problem 4. CNN embeddings usually separate image classes better than PCA applied to raw pixels. Give two reasons for this improvement.
Ans. Two key reasons are:
a. Task-driven, nonlinear feature learning
CNN embeddings are learned end-to-end using class labels, so they emphasize features that are discriminative for the task (e.g., shapes, parts). In contrast, PCA is linear and unsupervised—it only preserves directions of maximum variance, not class separability.
b. Invariance and hierarchical abstraction
CNNs build hierarchical features that are increasingly invariant to nuisance factors like translation, illumination, and small deformations. This reduces within-class variation and increases between-class separation, while raw-pixel PCA is highly sensitive to such variations.

Problem 5. YOLO predicts bounding boxes and class probabilities directly from convolutional feature maps. Explain why learning spatially-aware embeddings is essential for object detection, and why a pipeline based only on global image embeddings (e.g., PCA + classifier) would fail.
Ans. Object detection requires predicting what an object is and where it is in the image. CNN feature maps preserve spatial structure, so each location in the feature map corresponds to a specific region of the input image. This allows models like YOLO to associate object features with precise positions and predict bounding boxes relative to those positions. Why global embeddings are insufficient
A pipeline based on global embeddings (e.g., PCA on raw pixels followed by a classifier) collapses the entire image into a single vector. This causes several failures:
a. Loss of location information
Global embeddings discard spatial layout, so there is no way to infer object positions or sizes.
b. Inability to detect multiple objects
Multiple instances of the same class in different locations become indistinguishable when represented by one vector.
c. Background dominates the representation
PCA preserves directions of maximum variance, which are often due to background, lighting, or texture rather than object structure.
