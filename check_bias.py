
import cv2
import numpy as np
import tensorflow as tf
import json
import os

MODEL_PATH = "palm_recognition_model.h5"
CLASS_PATH = "class_names.json"
IMG_SIZE = 128

def check_bias():
    if not os.path.exists(MODEL_PATH):
        print("Model not found")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_PATH, 'r') as f:
        categories = json.load(f)
    
    print(f"Classes: {categories}")

    # 1. Random Noise
    noise = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype("float32")
    pred_noise = model.predict(noise, verbose=0)[0]
    idx_noise = np.argmax(pred_noise)
    print(f"\n[Random Noise] Prediction: {categories[idx_noise]}")
    for k, v in zip(categories, pred_noise):
        print(f"  {k}: {v:.4f}")

    # 2. Black Image (Empty dark room)
    black = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32")
    pred_black = model.predict(black, verbose=0)[0]
    idx_black = np.argmax(pred_black)
    print(f"\n[Black Image] Prediction: {categories[idx_black]}")
    for k, v in zip(categories, pred_black):
        print(f"  {k}: {v:.4f}")

    # 3. White Image (Bright light / empty wall)
    white = np.ones((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32")
    pred_white = model.predict(white, verbose=0)[0]
    idx_white = np.argmax(pred_white)
    print(f"\n[White Image] Prediction: {categories[idx_white]}")
    for k, v in zip(categories, pred_white):
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    check_bias()
