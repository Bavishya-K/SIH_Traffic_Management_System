import cv2
from ultralytics import YOLO

# Load the YOLOv8 model for accident detection
model = YOLO("../models/yolov8s_accident.pt")  # Adjust to your accident detection model path

# Confidence threshold for detecting accidents
CONFIDENCE_THRESHOLD = 0.9

def detect_accident_at_second(video_path, target_second):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return False

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
    target_frame = target_second * fps  # Calculate the frame corresponding to the target second

    # Ensure the target frame is within the video length
    if target_frame >= total_frames:
        print("Error: Specified second exceeds the video length.")
        cap.release()
        return False

    # Set the video to the target frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    accident_detected = False

    # Read the frame at the specified second
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read the frame at the specified second.")
        cap.release()
        return False

    # Run inference on the frame
    results = model(frame)

    # Check for high-confidence accident detections
    confidences = results[0].boxes.conf  # Confidence scores
    class_ids = results[0].boxes.cls  # Class IDs

    for conf, class_id in zip(confidences, class_ids):
        class_name = results[0].names[int(class_id)]
        if class_name == "accident" and conf >= CONFIDENCE_THRESHOLD:
            accident_detected = True

            # Draw bounding box and label
            boxes = results[0].boxes.xywh  # Bounding box coordinates (xywh format)
            for box in boxes:
                x, y, w, h = box
                x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for accident
                label = f"Accident: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame
    # cv2.imshow("Accident Detection", frame)

    # # Wait for a key press to close the display
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

    cap.release()
    return accident_detected
