import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import os

# --- CONFIGURATION ---
MODEL_PATH = 'emotion_model_SGD.h5'  # Make sure this matches your file name
DATA_X_PATH = 'X_emotion.npy'
DATA_Y_PATH = 'y_emotion.npy'
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def plot_confusion_matrix():
    # 1. Load Data
    if not os.path.exists(DATA_X_PATH):
        print("Error: X_emotion.npy not found. Please run preprocessing first.")
        return

    print("Loading data...")
    X = np.load(DATA_X_PATH)
    y = np.load(DATA_Y_PATH)

    # We need the Test set (the one the model hasn't seen mostly)
    # Using the same random_state=42 ensures we get the same split as training
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Load Model & Predict
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except:
        print("Error: Model file not found.")
        return

    print("Running predictions on test set...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 3. Generate Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # 4. Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTION_LABELS,
                yticklabels=EMOTION_LABELS)

    plt.title('Emotion Recognition Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot for your report
    plt.savefig('emotion_confusion_matrix.png', dpi=300)
    print("✅ Saved 'emotion_confusion_matrix.png'")
    plt.show()

    # 5. Print Classification Report (Precision/Recall numbers for table)
    print("\n--- Detailed Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=EMOTION_LABELS))


# Run the function
plot_confusion_matrix()