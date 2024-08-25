from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('arismodel.pt')  # Ensure this path is correct for your model

def initialize_video_capture():
    for index in range(5):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Video capture opened with index {index}")
            return cap
        cap.release()
    print("Error: No working video capture found.")
    return None

def main():
    cap = initialize_video_capture()
    if cap is None:
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break

        results = model(frame)

        annotated_frame = results[0].plot()  # Ensure this method is correct

        cv2.imshow('YOLO Live Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
