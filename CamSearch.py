import cv2


def list_cameras():
    print("Searching for cameras...")
    available_cameras = []

    # Check the first 5 indexes (0 to 4)
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Camera found at index {i}")
            available_cameras.append(i)
            cap.release()
        else:
            print(f"❌ No camera at index {i}")

    return available_cameras


cameras = list_cameras()

if not cameras:
    print("\n⚠️ No cameras detected! Check your USB connection.")
else:
    print(f"\n🎉 Found {len(cameras)} camera(s). Use index {cameras[0]} in your code.")