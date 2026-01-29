#!/usr/bin/env python
# coding: utf-8

# # Palm Recognition AI Project Report
# 
# ## Dataset Design and Class Balance
# The dataset consists of 5 classes:
# - **James**: 86 images
# - **Person_2**: 12 images
# - **Person_3**: 13 images
# - **Person_4**: 8 images
# - **Unknown**: 7 images
# 
# **Justification:** The dataset is currently **highly imbalanced**, with 'James' being the dominant class. This reflects a real-world scenario where the primary user is present most often. However, for robust training, we use data augmentation (resizing/normalization) and validaton splits. 
# 
# ## Image Size and Preprocessing
# - **Image Size**: 128x128 pixels.
# - **Justification**: This size mitigates computational cost while retaining sufficient spatial detail for palm feature extraction. Lower resolutions might lose texture details (lines), while higher resolutions would slow down real-time inference.
# - **Normalization**: Pixel values are scaled to [0, 1] to ensure model stability.
# 
# ## Ethical and Privacy Considerations
# - **Bias**: The dataset currently has limited diversity in skin tones and lighting conditions. This may lead to biased performance against underrepresented groups.
# - **Privacy**: Face data is not explicitly captured, but palm prints are biometric data. In a production system, these images should be encrypted.
# - **Live Testing**: The system includes a real-time detection module.
# 
# ## Evaluation Metrics
# - **Primary Metric**: Accuracy (overall correctness).
# - **Secondary Metrics**: Confusion Matrix and Precision/Recall are used to detect if the model is just predicting the majority class ("James").
# 
# ## Unknown Palm Detection
# - A confidence threshold (e.g., 0.8) is applied. If the maximum probability is below this, the prediction is rejected as "Unknown".

# In[ ]:


# Run this cell to install the necessary libraries into your current kernel
get_ipython().run_line_magic('pip', 'install opencv-python numpy matplotlib tensorflow scikit-learn seaborn')


# In[ ]:


import os, json
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping


# In[ ]:


DATASET_PATH = "./data set"
IMG_SIZE = 128

# Read folder names that actually contain images
categories = []
for d in os.listdir(DATASET_PATH):
    p = os.path.join(DATASET_PATH, d)
    if os.path.isdir(p) and len(os.listdir(p)) > 0:
        categories.append(d)

# IMPORTANT: Stable order
categories = sorted(categories)

# Must include UNKNOWN to satisfy "unknown palm" requirement

# Save class order so live testing maps correctly
with open("class_names.json", "w") as f:
    json.dump(categories, f)

label_dict = {cat: i for i, cat in enumerate(categories)}
print("✅ Classes (fixed order):", categories)

data, labels = [], []

for cat in categories:
    folder = os.path.join(DATASET_PATH, cat)
    imgs = os.listdir(folder)
    print(f"{cat}: {len(imgs)} images")

    for img_name in imgs:
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0
        data.append(img)
        labels.append(label_dict[cat])

data = np.array(data, dtype="float32")
labels = np.array(labels, dtype="int32")

X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)

y_train = to_categorical(y_train_raw, num_classes=len(categories))
y_test  = to_categorical(y_test_raw,  num_classes=len(categories))

print("✅ X_train:", X_train.shape, "X_test:", X_test.shape)


# In[ ]:


dataset_path = './data set'
IMG_SIZE = 128

# Filter only folders that contain images
valid_categories = []
if os.path.exists(dataset_path):
    for entry in os.listdir(dataset_path):
        entry_path = os.path.join(dataset_path, entry)
        if os.path.isdir(entry_path) and len(os.listdir(entry_path)) > 0:
            valid_categories.append(entry)
else:
    raise FileNotFoundError("Dataset directory not found! Make sure './data set' exists.")

# IMPORTANT: Stable class order (prevents wrong label mapping in live testing)
categories = sorted(valid_categories)

print(f"Found categories (sorted): {categories}")

if not categories:
    raise ValueError("No data found! Please run the Data Collection cell above.")


# Save class order for later (live camera cell must load this)
with open("class_names.json", "w") as f:
    json.dump(categories, f)

label_dict = {category: i for i, category in enumerate(categories)}

data = []
labels = []

print("Loading data and checking class balance...")
for category in categories:
    folder_path = os.path.join(dataset_path, category)
    images = os.listdir(folder_path)
    print(f"Class '{category}': {len(images)} images")

    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0  # Normalize
        data.append(img)
        labels.append(label_dict[category])

data = np.array(data, dtype="float32")
labels = np.array(labels, dtype="int32")

print(f"Data loaded: {data.shape}")
print(f"Labels loaded: {labels.shape}")

# Train/test split (stratified)
X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)

# One-hot encoding
y_train = to_categorical(y_train_raw, num_classes=len(categories))
y_test  = to_categorical(y_test_raw,  num_classes=len(categories))

print("✅ X_train:", X_train.shape, "X_test:", X_test.shape)
print("✅ y_train:", y_train.shape, "y_test:", y_test.shape)
print("✅ Classes saved to class_names.json")


# In[ ]:


# --- Model Building (FIXED: better stability + matches categories count) ---
if 'categories' in locals() and len(categories) > 0:
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(len(categories), activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
else:
    raise ValueError("Cannot build model - no categories found. Run preprocessing first.")


# In[ ]:


# --- Training (FIXED: Augmentation + EarlyStopping + Save model) ---
if 'X_train' in locals():
    # Data augmentation to improve generalisation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.10,
        brightness_range=(0.8, 1.2),
        horizontal_flip=False
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=25,
        validation_data=(X_test, y_test),
        callbacks=[early_stop]
    )

    model.save('palm_recognition_model.h5')
    print("✅ Model saved as palm_recognition_model.h5")

    # Plot Training History
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()
else:
    raise ValueError("X_train not found. Run preprocessing first.")


# In[ ]:


# --- Evaluation & Confusion Matrix (FIXED: no seaborn needed) ---
if 'X_test' in locals():
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Classification Report
    print("Classification Report:\n")
    print(classification_report(y_true, y_pred_classes, target_names=categories))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(categories)), categories, rotation=45, ha="right")
    plt.yticks(range(len(categories)), categories)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
else:
    raise ValueError("X_test not found. Run preprocessing + training first.")


# In[ ]:


# --- Real-time Prediction (FIXED: loads class_names.json + UNKNOWN class + threshold backup) ---
import os, json
import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.70  # backup rejection rule

MODEL_PATH = "palm_recognition_model.h5"
CLASS_PATH = "class_names.json"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found. Please run training first.")

if not os.path.exists(CLASS_PATH):
    raise FileNotFoundError("class_names.json not found. Run preprocessing cell that saves class order.")

# Load model + stable class list
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    categories = json.load(f)

# Ensure UNKNOWN exists (recommended for the brief)
if "UNKNOWN" not in categories:
    print("⚠️ WARNING: 'UNKNOWN' class not found in categories. Add ./data set/UNKNOWN with images for best compliance.")

category_map = {i: cat for i, cat in enumerate(categories)}
print("✅ Loaded classes:", categories)

cap = cv2.VideoCapture(0)
print("Starting Real-time Recognition... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, axis=0).astype("float32") / 255.0

    # Predict
    pred = model.predict(img, verbose=0)[0]
    class_idx = int(np.argmax(pred))
    confidence = float(np.max(pred))
    predicted_label = category_map.get(class_idx, "UNKNOWN")

    # Unknown decision:
    # 1) If model predicts UNKNOWN class -> Unknown
    # 2) Else if confidence below threshold -> Unknown (backup)
    if predicted_label == "UNKNOWN" or confidence < CONFIDENCE_THRESHOLD:
        label = "Unknown"
        color = (0, 0, 255)  # Red
    else:
        label = predicted_label
        color = (0, 255, 0)  # Green

    text = f"{label}: {confidence:.2f}"
    cv2.putText(frame, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Palm Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

