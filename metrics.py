from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from ultralytics import YOLO
# Example: Replace with your test ground truth labels and predictions
ground_truth = ['Anxious', 'Depressed', 'Normal', 'Anxious', 'Normal']  # Replace with actual ground truth labels
predictions = ['Anxious', 'Depressed', 'Normal', 'Normal', 'Normal'] 

# Calculate classification metrics
print("\n Classification Report:")
print(classification_report(ground_truth, predictions))

# Calculate confusion matrix
cm = confusion_matrix(ground_truth, predictions, labels=['Normal', 'Depressed', 'Anxious'])
df_cm = pd.DataFrame(cm, index=['Normal', 'Depressed', 'Anxious'], columns=['Normal', 'Depressed', 'Anxious'])

print("\n Confusion Matrix:")
print(df_cm)


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(ground_truth, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Load the trained model
model = YOLO("best.pt")

# Run evaluation
metrics = model.val(data="E:/CSM/4-2/MAJOR PROJECT/Facial Data.v2i.yolov11/data.yaml")

# Print evaluation metrics
print("Detection Metrics:")
print(f"Precision: {metrics.box.mp:.3f}")
print(f"Recall: {metrics.box.mr:.3f}")
print(f"mAP@0.5: {metrics.box.map50:.3f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")