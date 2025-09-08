import cv2
from ultralytics import YOLO
from tracker import Tracker

# Initialize YOLO model
model = YOLO('../models/yolov8m.pt')

# Function to process video and return in/out counts at a specific second
def get_vehicle_counts_at_second(video_path, target_second=20):
    tracker = Tracker()
    cap = cv2.VideoCapture(video_path)

    # Check if video is opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return 0, 0

    # Get video frame rate and calculate the target frame number at the specified second
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    target_frame = target_second * fps  # Convert seconds to frame number

    in_count, out_count = 0, 0
    in_ids, out_ids = set(), set()

    # Set the video to the target frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    # Read the frame at the specified second
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read the frame at second {target_second}. Returning counts as (0, 0).")
        cap.release()
        return 0, 0  # Return 0 if frame can't be read

    # Resize frame for consistent processing
    resized_width, resized_height = 1020, 500
    frame = cv2.resize(frame, (resized_width, resized_height))

    # Define the vertical counting line (placed at the center of the resized frame width)
    vertical_line_x = resized_width // 2

    # Run YOLO detection
    results = model.predict(frame)
    objects_rects = []

    for result in results:
        for box in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, class_id = map(float, box[:6])
            objects_rects.append([x1, y1, x2 - x1, y2 - y1])

    # Update tracker with detected objects
    bbox_ids = tracker.update(objects_rects)

    for bbox in bbox_ids:
        x, y, w, h, obj_id = bbox
        cx, cy = x + w // 2, y + h // 2  # Calculate the center of the bounding box

        # Determine whether the object is on the left (outbound) or right (inbound) of the line
        if cx < vertical_line_x:
            if obj_id not in out_ids:  # Outbound vehicle
                out_count += 1
                out_ids.add(obj_id)
        else:
            if obj_id not in in_ids:  # Inbound vehicle
                in_count += 1
                in_ids.add(obj_id)

        # Draw the bounding box and vertical line for visualization (optional)
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {obj_id}", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw the vertical counting line
    cv2.line(frame, (vertical_line_x, 0), (vertical_line_x, resized_height), (255, 0, 0), 2)

    # Optionally, display the processed frame (for testing purposes)
    cv2.imshow("Vehicle Detection at Target Second", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()

    # Return valid counts (even if no vehicles are detected)
    return in_count, out_count