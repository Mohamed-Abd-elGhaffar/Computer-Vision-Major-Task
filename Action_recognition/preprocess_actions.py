import cv2  # OpenCV
import numpy as np
import os

# --- CONFIGURATION ---
# Change this to where your UCF-101 folder is
DATA_PATH = "E:/GradProj/Data_Gathering/PythonProject/Action_recognition/UCF-101"

# We will start with just 3 classes to test if it works.
# (You can add more later)
CLASSES = ["Typing", "Hammering", "WallPushups", "WritingOnBoard"]

IMG_SIZE = 64  # Resize all frames to 64x64
SEQ_LENGTH = 10  # Take 10 frames per video


def extract_frames(video_path):
    """Opens a video and returns a sequence of 10 frames."""
    cap = cv2.VideoCapture(video_path)

    # Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate which frames to grab (e.g., frame 0, 10, 20...)
    # This ensures we cover the WHOLE action, not just the first second.
    skip = max(int(total_frames / SEQ_LENGTH), 1)

    frames = []
    for i in range(SEQ_LENGTH):
        # Set the camera position to specific frame ID
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip)
        ret, frame = cap.read()

        if ret:
            # Resize and Normalize (0-1)
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame / 255.0  # Normalize pixel values
            frames.append(frame)
        else:
            # If video is too short, pad with zeros
            frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))

    cap.release()
    return np.array(frames)


def create_dataset():
    X = []  # Data
    y = []  # Labels

    print("Starting data loading...")

    for label_index, class_name in enumerate(CLASSES):
        folder_path = os.path.join(DATA_PATH, class_name)

        # Check if folder exists
        if not os.path.exists(folder_path):
            print(f"Error: Folder not found: {folder_path}")
            continue

        print(f"Processing class: {class_name}")

        # Loop through every video in that folder
        for video_file in os.listdir(folder_path):
            video_path = os.path.join(folder_path, video_file)

            # Extract frames
            sequence = extract_frames(video_path)

            X.append(sequence)
            y.append(label_index)  # 0, 1, or 2

    X = np.array(X)
    y = np.array(y)

    print("\n--- Data Loading Complete ---")
    print(f"X Shape: {X.shape}")  # Should be (Num_Videos, 10, 64, 64, 3)
    print(f"y Shape: {y.shape}")

    return X, y


# Run the function
if __name__ == "__main__":
    X_train, y_train = create_dataset()

    # Save to file so we don't have to wait next time
    np.save("X_data.npy", X_train)
    np.save("y_data.npy", y_train)
    print("Data saved to .npy files!")