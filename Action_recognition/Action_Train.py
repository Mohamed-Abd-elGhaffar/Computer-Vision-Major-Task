import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Flatten, Input, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD, Adagrad
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time

# --- 1. Load Data ---
print("Loading data...")
X = np.load("X_data.npy")
y = np.load("y_data.npy")

# --- 2. Correct 3-Way Split ---
# First, separate Training (70%) from the rest
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)

# Second, split the Rest into Validation (15%) and Test (15%)
# Note: We split X_temp, NOT X!
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

print(f"Training shapes:   {X_train.shape}")
print(f"Validation shapes: {X_val.shape}")
print(f"Test shapes:       {X_test.shape}")


# --- 3. Build Model Function ---
def build_model():
    # Define Base
    base_cnn = MobileNetV2(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    base_cnn.trainable = False  # Start frozen

    model = Sequential([
        Input(shape=(10, 64, 64, 3)),
        TimeDistributed(base_cnn),  # Layer 0 in the sequential list
        TimeDistributed(Flatten()),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dropout(0.3),
        Dense(4, activation='softmax')
    ])

    # RETURN BOTH the full model AND the base_cnn so we can unfreeze it later
    return model, base_cnn


# ... [Step 1, 2, 3 (Load Data, Split, Build Model) remain the same] ...

# --- 4. The Experiment Loop (Corrected) ---

# We define a config so Phase 2 uses the SAME optimizer class but a smaller LR
optimizer_configs = {
    'Adam': {
        'class': Adam,
        'lr_phase1': 0.001,
        'lr_phase2': 0.00001  # 1e-5 (Adam needs to be very slow for fine-tuning)
    },
    'SGD': {
        'class': SGD,
        'lr_phase1': 0.01,  # Increased to 0.01 so it learns faster
        'lr_phase2': 0.0001  # 1e-4 (SGD is safer, so we can go slightly faster than Adam)
    },
    'Adagrad': {
        'class': Adagrad,
        'lr_phase1': 0.01,
        'lr_phase2': 0.0001  # 1e-4
    }
}

results = {}
test_scores1 = {}
test_scores2 = {}
for opt_name, config in optimizer_configs.items():
    print(f"\n--- Training with {opt_name} ---")
    tf.keras.backend.clear_session()

    # Get fresh model
    model, base_cnn = build_model()

    # --- PHASE 1: Training Top Layers (Frozen Base) ---
    print(f"Phase 1: Training with {opt_name} (LR: {config['lr_phase1']})...")

    # Instantiate the optimizer for Phase 1
    opt_phase1 = config['class'](learning_rate=config['lr_phase1'])

    model.compile(optimizer=opt_phase1,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    start_time = time.time()
    history1 = model.fit(X_train, y_train,
                         epochs=20,
                         batch_size=8,
                         validation_data=(X_val, y_val),
                         verbose=1)
    # --- TEST EVALUATION ---
    print(f"Evaluating {opt_name} on Unseen Test Set...")
    test_loss1, test_acc1 = model.evaluate(X_test, y_test, verbose=0)
    test_scores1[opt_name] = test_acc1 * 100
    print(f"{opt_name} Final Test Accuracy: {test_acc1 * 100:.2f}%")

    filename = f"action_model_Frozen_{opt_name}.h5"
    model.save(filename)

    # --- PHASE 2: Fine-Tuning (Unfrozen Base) ---
    print(f"Phase 2: Unfreezing and Fine-Tuning with {opt_name} (LR: {config['lr_phase2']})...")
    base_cnn.trainable = True

    # Instantiate the SAME optimizer type for Phase 2, but with lower LR
    opt_phase2 = config['class'](learning_rate=config['lr_phase2'])

    # Recompile to apply the new learning rate and trainable weights
    model.compile(optimizer=opt_phase2,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history2 = model.fit(X_train, y_train,
                         epochs=10,
                         batch_size=8,
                         validation_data=(X_val, y_val),
                         verbose=1)

    end_time = time.time()

    # --- MERGE HISTORIES ---
    # Combine the lists so the graph is one continuous line
    full_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    full_val_loss = history1.history['val_loss'] + history2.history['val_loss']

    results[opt_name] = {
        'val_accuracy': full_val_acc,
        'val_loss': full_val_loss,
        'time': end_time - start_time
    }

    # --- TEST EVALUATION ---
    print(f"Evaluating {opt_name} on Unseen Test Set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    test_scores2[opt_name] = test_acc * 100
    print(f"{opt_name} Final Test Accuracy: {test_acc * 100:.2f}%")
    print(f"{opt_name} Final Test Loss: {test_loss * 100:.2f}%")
    print(f"{opt_name} Total time: {end_time - start_time:.2f}s")
    # --- saving model ---
    filename = f"action_model_Trained_{opt_name}.h5"
    model.save(filename)

# ... [Step 5: Plotting remains the same] ...

# --- 5. Plotting (Line + Bar Chart) ---
plt.figure(figsize=(18, 5))

# Plot 1: Validation Accuracy (Line)
plt.subplot(2, 2, 1)
for name, res in results.items():
    plt.plot(res['val_accuracy'], label=name)
# Draw a vertical line where fine-tuning started
plt.axvline(x=20, color='gray', linestyle='--', label='Start Fine-Tuning')
plt.title('Validation Accuracy (Phase 1 + 2)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot 2: Validation Loss (Line)
plt.subplot(2, 2, 2)
for name, res in results.items():
    plt.plot(res['val_loss'], label=name)
plt.axvline(x=20, color='gray', linestyle='--')
plt.title('Validation Loss (Phase 1 + 2)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot 3: Final Test Scores (Bar Chart)
plt.subplot(2, 2, 3)
names = list(test_scores1.keys())
values = list(test_scores1.values())
bars = plt.bar(names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])  # Blue, Orange, Green
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.1f}%', ha='center', va='bottom')
plt.title('Final Test Accuracy (Not Trainable)')

plt.subplot(2, 2, 4)
names = list(test_scores2.keys())
values = list(test_scores2.values())
bars = plt.bar(names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])  # Blue, Orange, Green
# Add numbers on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.1f}%', ha='center', va='bottom')

plt.title('Final Test Accuracy (After Training)')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)

plt.tight_layout()
plt.show()