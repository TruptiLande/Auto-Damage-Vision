1. What changes during training?
Weight Optimization: The model adjusts its internal parameters to minimize error.
Loss Reduction: box_loss (localization) and cls_loss (classification) trend downward.
Metric Improvement: Accuracy scores like mAP50, Precision, and Recall increase as the model learns specific damage patterns rather than general objects.
Feature Specialization: The model shifts from recognizing general shapes to identifying specific pixel patterns for dents (mop_lom) and scratches (tray_son).

2. Why might training be unstable?
Small Dataset: Using only 160 images can lead to overfitting, where the model memorizes specific images instead of learning general traits.
Learning Rate (LR): If too high, the model "overshoots" the optimal weights, causing the loss to fluctuate.
Batch Size: Small batches (like batch=8 in your code) can introduce "noise," making weight updates erratic.
Class Imbalance: If one damage type is rarer than others, the model may struggle to learn it consistently.
