import numpy as np
from traffic_count_detection import get_vehicle_counts_at_second
from emergency_detection import detect_emergency_at_second
from accident_detection import detect_accident_at_second

def get_inputs():
    # Get in/out counts from four different video streams
    in_counts = []
    out_counts = []

    # Define video paths
    video_paths = [
        '../videos/count/count_2.mp4',
        '../videos/count/count_4.mp4',
        '../videos/count/count_6.mp4',
        '../videos/count/count_7.mp4'
    ]

    # Initialize flags
    emergency_detected = False
    accident_detected = False

    # Define the target second for detection
    target_second = 10  # Specify the second for emergency and accident detection

    for path in video_paths:
        # Get vehicle counts
        in_count, out_count = get_vehicle_counts_at_second(path, target_second=20)
        in_counts.append(in_count)
        out_counts.append(out_count)

        # Check for emergency vehicles at the specified second
        if detect_emergency_at_second(path, target_second):
            emergency_detected = True

        # Check for accidents at the specified second
        if detect_accident_at_second(path, target_second):
            accident_detected = True

    # Set flags based on detection results
    emergency = 1 if emergency_detected else 0
    accident = 1 if accident_detected else 0

    # Calculate queue length
    queue_length = sum(in_counts) - sum(out_counts)  # Approximate queue

    # Placeholder neighbor state (modify as needed based on your use case)
    neighbor_state = [0] * 8

    # Print results for debugging
    print(f"In Counts: {in_counts}")
    print(f"Out Counts: {out_counts}")
    print(f"Emergency Detected: {emergency}")
    print(f"Accident Detected: {accident}")
    print(f"Queue Length: {queue_length}")
    print(f"Neighbor State: {neighbor_state}")

    return in_counts, out_counts, emergency, accident, queue_length, neighbor_state


# Example usage
if __name__ == "__main__":
    in_counts, out_counts, emergency, accident, queue_length, neighbor_state = get_inputs()
