
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

DATASET_PATH = r"c:\Users\alanb\OneDrive\Desktop\palm_recognization_ai\data set"
TARGET_COUNT = 100
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.7, 1.3),
    fill_mode='nearest'
)

def augment_class(class_name):
    class_path = os.path.join(DATASET_PATH, class_name)
    if not os.path.isdir(class_path):
        return

    files = [f for f in os.listdir(class_path) if f.lower().endswith(IMG_EXTS)]
    count = len(files)
    
    if count >= TARGET_COUNT:
        print(f"✅ {class_name}: {count} images (sufficient)")
        return

    needed = TARGET_COUNT - count
    print(f"🔄 {class_name}: {count} images. Generating {needed} more...")

    # Load existing images
    images = []
    for fname in files:
        img = load_img(os.path.join(class_path, fname))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        images.append(x)

    if not images:
        print(f"⚠️  {class_name} is empty! Cannot augment.")
        return

    generated = 0
    # Cycle through existing images and generate new ones
    while generated < needed:
        for img in images:
            if generated >= needed:
                break
            
            # Generate one batch (1 image)
            for batch in datagen.flow(img, batch_size=1, save_to_dir=class_path, save_prefix='aug', save_format='jpg'):
                generated += 1
                break # Only 1 user per flow call

    print(f"   Generated {generated} images. New total: {count + generated}")

def main():
    if not os.path.exists(DATASET_PATH):
        print("Dataset path not found.")
        return

    categories = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
    for cat in categories:
        augment_class(cat)

if __name__ == "__main__":
    main()
