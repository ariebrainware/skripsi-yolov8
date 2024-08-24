from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLOv8 model
model = YOLO('arismodel.pt')  # Ensure this path is correct for your model

def initialize_video_capture():
    # Try different indices to find a working webcam
    for index in range(5):  # Adjust the range if you have more cameras
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Video capture opened with index {index}")
            return cap
        cap.release()
    print("Error: No working video capture found.")
    return None

def main():
    # Initialize video capture
    cap = initialize_video_capture()
    if cap is None:
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break

        # Perform object detection
        results = model(frame)

        # Render the results on the frame
        annotated_frame = results[0].plot()  # Ensure this method is correct

        # Display the annotated frame
        cv2.imshow('YOLO Live Detection', annotated_frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
