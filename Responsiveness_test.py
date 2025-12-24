import time
import numpy as np
import tensorflow as tf

def measure_responsiveness(model_path, input_shape):
    print(f"--- Testing Latency for {model_path} ---")
    try:
        model = tf.keras.models.load_model(model_path)
    except:
        print("Model not found.")
        return

    # Create dummy data (1 sample)
    dummy_input = np.random.random(input_shape).astype(np.float32)

    # Warmup (GPU/CPU needs to wake up)
    for _ in range(5):
        model.predict(dummy_input, verbose=0)

    # Measure real speed
    start_time = time.time()
    iterations = 100
    for _ in range(iterations):
        model.predict(dummy_input, verbose=0)
    end_time = time.time()

    avg_time = (end_time - start_time) / iterations
    print(f"Average Inference Time: {avg_time*1000:.2f} ms")
    print(f"Max Possible FPS: {1/avg_time:.2f} FPS")

# 1. Test Action Model Responsiveness
# Shape: (1, 10, 64, 64, 3) -> 1 batch, 10 frames, 64x64 RGB
measure_responsiveness('./Action_recognition/action_model_Trained_SGD.h5', (1, 10, 64, 64, 3))

# 2. Test Emotion Model Responsiveness
# Shape: (1, 48, 48, 1) -> 1 batch, 48x48 Grayscale
measure_responsiveness('./emotion_recognition/emotion_model_SGD.h5', (1, 48, 48, 1))