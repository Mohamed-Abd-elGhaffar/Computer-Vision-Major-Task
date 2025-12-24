import os
import cv2
import numpy as np
from tqdm import tqdm  # Progress bar

# --- CONFIGURATION ---
DATA_DIR = './train'
TEST_DIR = './test'
IMG_SIZE = 48
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

X = []
y = []
X_test = []
y_test = []

print("Reading images and converting to .npy...")
#training images
for class_id, class_name in enumerate(CLASSES):
    path = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(path):
        print(f"Warning: Folder '{class_name}' not found. Skipping.")
        continue

    print(f"Processing class: {class_name} ({class_id})")

    # Iterate over all images in the folder
    for img_name in tqdm(os.listdir(path)):
        try:
            img_path = os.path.join(path, img_name)

            # 1. Read Image in Grayscale (0 means gray)
            img_array = cv2.imread(img_path, 0)

            # 2. Resize to ensure 48x48
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

            # 3. Add to list
            X.append(new_array)
            y.append(class_id)

        except Exception as e:
            pass
#Testing images
for class_id, class_name in enumerate(CLASSES):
    path = os.path.join(TEST_DIR, class_name)
    if not os.path.exists(path):
        print(f"Warning: Folder '{class_name}' not found. Skipping.")
        continue

    print(f"Processing class: {class_name} ({class_id})")

    # Iterate over all images in the folder
    for img_name in tqdm(os.listdir(path)):
        try:
            img_path = os.path.join(path, img_name)

            # 1. Read Image in Grayscale (0 means gray)
            img_array = cv2.imread(img_path, 0)

            # 2. Resize to ensure 48x48
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

            # 3. Add to list
            X_test.append(new_array)
            y_test.append(class_id)

        except Exception as e:
            pass

# --- CONVERT TO NUMPY AND SAVE ---
print("Reshaping and Saving...")

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # (N, 48, 48, 1)
y = np.array(y)
X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # (N, 48, 48, 1)
y_test = np.array(y_test)

# Normalize pixel values (0-255 -> 0-1) to save space/time later
X = X / 255.0
X_test = X_test / 255.0


np.save('X_emotion.npy', X)
np.save('y_emotion.npy', y)
np.save('X_emotion_test.npy', X_test)
np.save('y_emotion_test.npy', y_test)

print(f"Done! Saved {len(X)+len(X_test)} images.")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y shape: {y_test.shape}")