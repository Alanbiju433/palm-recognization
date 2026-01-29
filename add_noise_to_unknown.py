
import os
import cv2
import numpy as np

DATASET_PATH = r"c:\Users\alanb\OneDrive\Desktop\palm_recognization_ai\data set"
UNKNOWN_PATH = os.path.join(DATASET_PATH, "UNKNOWN")
IMG_SIZE = 128
COUNT = 100

def add_noise():
    if not os.path.exists(UNKNOWN_PATH):
        os.makedirs(UNKNOWN_PATH)
        
    print(f"Generating {COUNT} noise images in {UNKNOWN_PATH}...")
    
    for i in range(COUNT):
        # 1. Random Noise
        noise = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype="uint8")
        cv2.imwrite(os.path.join(UNKNOWN_PATH, f"noise_{i}.jpg"), noise)
        
        # 2. Black
        black = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype="uint8")
        cv2.imwrite(os.path.join(UNKNOWN_PATH, f"black_{i}.jpg"), black)
        
        # 3. White
        white = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype="uint8") * 255
        cv2.imwrite(os.path.join(UNKNOWN_PATH, f"white_{i}.jpg"), white)
        
        # 4. Gray
        gray = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype="uint8") * 127
        cv2.imwrite(os.path.join(UNKNOWN_PATH, f"gray_{i}.jpg"), gray)

    print("Done.")

if __name__ == "__main__":
    add_noise()
