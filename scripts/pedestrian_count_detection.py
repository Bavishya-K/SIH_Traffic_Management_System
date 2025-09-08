import cv2
from ultralytics import YOLO
from tracker import Tracker

# Initialize YOLO model
model = YOLO('../models/yolov8m.pt')

# Function to process video and return pedestrian counts at a specific second
def get_pedestrian_counts_at_second(video_path, target_second=20):
    tracker = Tracker()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return 0, 0

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    target_frame = target_second * fps
    in_count, out_count = 0, 0
    in_ids, out_ids = set(), set()

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read the frame at second {target_second}. Returning counts as (0, 0).")
        cap.release()
        return 0, 0

    resized_width, resized_height = 1020, 500
    frame = cv2.resize(frame, (resized_width, resized_height))

    # Define the vertical counting lines for pedestrian crossing areas (closer to left and right)
    vertical_line_x_left = int(resized_width * 0.1)  # 20% from the left side of the frame
    vertical_line_x_right = int(resized_width * 0.9)  # 80% from the left side of the frame

    results = model.predict(frame)
    objects_rects = []

    for result in results:
        for box in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, class_id = map(float, box[:6])
            # If detected class is pedestrian (class_id of pedestrian in YOLO is typically 0 or 1)
            if class_id == 0:  # Change based on the class ID for pedestrians in your model
                objects_rects.append([x1, y1, x2 - x1, y2 - y1])

    bbox_ids = tracker.update(objects_rects)

    for bbox in bbox_ids:
        x, y, w, h, obj_id = bbox
        cx, cy = x + w // 2, y + h // 2  # Calculate the center of the bounding box

        # Check if pedestrian is on the left or right side of the pedestrian crossing area
        if cx < vertical_line_x_left:
            if obj_id not in out_ids:  # Pedestrian crossing from left
                out_count += 1
                out_ids.add(obj_id)
        elif cx > vertical_line_x_right:
            if obj_id not in in_ids:  # Pedestrian crossing from right
                in_count += 1
                in_ids.add(obj_id)

        # Draw bounding box and vertical lines for visualization
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {obj_id}", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw vertical lines for left and right pedestrian crossing areas
    cv2.line(frame, (vertical_line_x_left, 0), (vertical_line_x_left, resized_height), (255, 0, 0), 2)
    cv2.line(frame, (vertical_line_x_right, 0), (vertical_line_x_right, resized_height), (255, 0, 0), 2)

    # Optionally, display the processed frame
    cv2.imshow("Pedestrian Detection at Target Second", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()

    # Return counts
    return in_count, out_count