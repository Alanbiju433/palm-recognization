
import os
import cv2
import numpy as np
import tensorflow as tf
import json
from sklearn.metrics import classification_report, confusion_matrix

DATASET_PATH = r"c:\Users\alanb\OneDrive\Desktop\palm_recognization_ai\data set"
MODEL_PATH = "palm_recognition_model.h5"
CLASS_PATH = "class_names.json"
IMG_SIZE = 128
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def evaluate():
    if not os.path.exists(MODEL_PATH):
        print("Model not found")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_PATH, 'r') as f:
        categories = json.load(f)
    
    print(f"Classes: {categories}")
    
    y_true = []
    y_pred = []
    
    label_map = {cat: i for i, cat in enumerate(categories)}

    print("Loading all images...")
    for cat in categories:
        path = os.path.join(DATASET_PATH, cat)
        if not os.path.isdir(path):
            continue
            
        class_idx = label_map[cat]
        files = [f for f in os.listdir(path) if f.lower().endswith(IMG_EXTS)]
        
        for fname in files:
            img_path = os.path.join(path, fname)
            img = cv2.imread(img_path)
            if img is None: continue
            
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)
            
            pred = model.predict(img, verbose=0)[0]
            pred_idx = np.argmax(pred)
            
            y_true.append(class_idx)
            y_pred.append(pred_idx)
            
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=categories))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

if __name__ == "__main__":
    evaluate()
