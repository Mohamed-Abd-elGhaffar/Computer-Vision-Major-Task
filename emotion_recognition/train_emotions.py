import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, SGD, Adagrad
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # <--- NEW IMPORTS
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time

# --- 1. LOAD DATA ---
print("Loading Emotion Data (.npy)...")
X = np.load('X_emotion.npy')
y = np.load('y_emotion.npy')
X_test = np.load('X_emotion_test.npy')
y_test = np.load('y_emotion_test.npy')
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
num_classes = len(np.unique(y))

# --- 2. DEFINE MODEL ---
def build_emotion_model():
    model = Sequential([
        Input(shape=(48, 48, 1)),

        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(256, (3, 3), activation='relu', padding='same'),  # Added extra layer for better learning
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Classifier
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),  # Keep 0.5 here to prevent overfitting
        Dense(num_classes, activation='softmax')
    ])
    return model


# --- 3. EXPERIMENT SETUP ---
optimizers = {
    # Start with slightly higher LR, let the callback lower it if needed
    'Adam': Adam(learning_rate=0.001),
    'SGD': SGD(learning_rate=0.01, momentum=0.9),  # Added momentum to help SGD
    'Adagrad': Adagrad(learning_rate=0.01)
}

results = {}
test_scores = {}

# Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# --- 4. TRAINING LOOP ---
for opt_name, optimizer in optimizers.items():
    print(f"\n--- Training Emotion Model with {opt_name} ---")
    tf.keras.backend.clear_session()

    model = build_emotion_model()
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # --- DEFINE CALLBACKS (The Automatic Tuners) ---
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
    ]

    start_time = time.time()

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=64),
        epochs=60,  # Increased max epochs (EarlyStopping will stop it earlier if needed)
        validation_data=(X_val, y_val),
        callbacks=callbacks,  # <--- Attach callbacks here
        verbose=1
    )

    end_time = time.time()

    results[opt_name] = {
        'val_accuracy': history.history['val_accuracy'],
        'val_loss': history.history['val_loss'],
        'time': end_time - start_time
    }

    print(f"Evaluating {opt_name} on Unseen Test Set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    test_scores[opt_name] = test_acc * 100
    print(f"{opt_name} Final Test Accuracy: {test_acc * 100:.2f}%")

    model.save(f"emotion_model_{opt_name}.h5")

# --- 5. PLOTTING ---
plt.figure(figsize=(18, 5))

# Accuracy
plt.subplot(1, 3, 1)
for name, res in results.items():
    plt.plot(res['val_accuracy'], label=name)
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.legend()

# Loss
plt.subplot(1, 3, 2)
for name, res in results.items():
    plt.plot(res['val_loss'], label=name)
plt.title('Validation Loss')
plt.xlabel('Epochs')
plt.legend()

# Bar Chart
plt.subplot(1, 3, 3)
names = list(test_scores.keys())
values = list(test_scores.values())
bars = plt.bar(names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.1f}%', ha='center', va='bottom')
plt.title('Final Test Accuracy')
plt.ylim(0, 100)

plt.tight_layout()
plt.show()