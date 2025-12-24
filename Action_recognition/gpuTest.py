import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

if tf.test.is_built_with_cuda():
    print("CUDA (NVIDIA software) is installed!")
else:
    print("CUDA is NOT found. You are likely using CPU.")