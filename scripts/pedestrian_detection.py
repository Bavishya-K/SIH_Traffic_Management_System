from ultralytics import YOLO
import cv2

# Load the YOLOv8m model pre-trained on COCO (which includes the 'person' class)
model = YOLO("yolov8m.pt")

def detect_pedestrians(video_path, target_second, confidence_threshold=0.5, show_output=True):
    """
    Detect pedestrians in the given video at a specific second using YOLOv8m and visualize the output.

    Parameters:
        video_path (str): Path to the video file.
        target_second (int): The second at which to analyze the frame.
        confidence_threshold (float): Minimum confidence score to consider a detection valid.
        show_output (bool): Whether to display the detection output.

    Returns:
        bool: True if pedestrians are detected, False otherwise.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second of the video
    frame_number = int(target_second * fps)  # Calculate the frame number to extract
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Jump to the target frame
    ret, frame = cap.read()

    if not ret:
        print(f"Error: Unable to read the frame at second {target_second}")
        cap.release()
        return False  # Return False if frame cannot be read

    # Perform pedestrian detection
    results = model(frame)  # YOLOv8 inference
    detections = results[0].boxes.data.cpu().numpy()  # Extract detections (bounding boxes, confidence, class IDs)
    
    person_detected = False
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        if int(class_id) == 0 and confidence >= confidence_threshold:  # Class ID 0 is 'person'
            person_detected = True
            # Draw the bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"Person: {confidence:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detections
    if show_output:
        cv2.imshow("Pedestrian Detection", frame)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()

    cap.release()  # Release video resource
    return person_detected