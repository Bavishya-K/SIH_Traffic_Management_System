import numpy as np
import tensorflow as tf
from collections import deque
import time
from traffic_count_detection import get_vehicle_counts_at_second
from emergency_detection import detect_emergency_at_second
from accident_detection import detect_accident_at_second

# Load the trained RL model
model = tf.keras.models.load_model("../models/rl_trained_model.keras")

# Phase encoding with four traffic phases
PHASE_ENCODING = {
    0: [1, 0, 0, 0],  # North-South Straight
    1: [0, 1, 0, 0],  # East-West Straight
    2: [0, 0, 1, 0],  # North-South Left Turn
    3: [0, 0, 0, 1],  # East-West Left Turn
}

def get_state(in_counts, out_counts, emergency, accident, queue_length, current_phase, phase_history, neighbor_state):
    in_counts = [count / 10.0 for count in in_counts]
    out_counts = [count / 10.0 for count in out_counts]
    neighbor_state = [count / 10.0 for count in neighbor_state]

    state = []
    state.extend(in_counts)
    state.extend(out_counts)
    state.extend([emergency, accident, queue_length])
    state.extend(PHASE_ENCODING[current_phase])
    state.extend(np.array(phase_history).flatten())  # Flatten phase history
    state.extend(neighbor_state)
    return np.array(state, dtype=np.float32)

def act(state, model):
    state = state.reshape(1, -1)
    q_values = model.predict(state, verbose=0)
    return np.argmax(q_values[0])

def get_inputs(target_second):
    # Define video paths
    video_paths = [
        '../videos/count/count_2.mp4',
        '../videos/count/count_4.mp4',
        '../videos/count/count_6.mp4',
        '../videos/count/count_7.mp4'
    ]
    # video_paths = [
    #     '../videos/cut/lane_1.mp4',
    #     '../videos/cut/lane_2.mp4',
    #     '../videos/cut/lane_3.mp4',
    #     '../videos/cut/lane_4.mp4'
    # ]

    # Initialize variables
    in_counts = []
    out_counts = []
    emergency_detected = False
    accident_detected = False

    for path in video_paths:
        # Get vehicle counts at the target second
        in_count, out_count = get_vehicle_counts_at_second(path, target_second)
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
    queue_length = max(0, sum(in_counts) - sum(out_counts))  # Ensure non-negative value

    # Placeholder neighbor state
    neighbor_state = [0] * 8  # Modify as needed for neighbor data

    return in_counts, out_counts, emergency, accident, queue_length, neighbor_state

def change_phase(phase):
    phases = [
        "North : Green (Remaining directions are Red)",
        "South : Green (Remaining directions are Red)",
        "East : Green (Remaining directions are Red)",
        "West : Green (Remaining directions are Red)"
    ]
    print(f"Changing to: {phases[phase]}")


def main_inference():
    # Track last 3 phases
    phase_history = deque([[0, 0, 0, 0] for _ in range(3)], maxlen=3)
    current_phase = 0  # Initialize with phase 0
    interval = 2  # Time interval in seconds to fetch data
    target_second = 2  # Initial second to analyze

    while True:
        # Get input data for the current target second
        in_counts, out_counts, emergency, accident, queue_length, neighbor_state = get_inputs(target_second)

        # Prepare the state for RL model
        state = get_state(
            in_counts, out_counts, emergency, accident, queue_length,
            current_phase, phase_history, neighbor_state
        )

        # Predict the next phase using the RL model
        next_phase = act(state, model)

        # Log and apply the new phase
        change_phase(next_phase)

        # Update phase history and set the current phase
        phase_history.append(PHASE_ENCODING[current_phase])
        current_phase = next_phase

        # Update the target second for the next interval
        target_second += interval

        # Wait for the next interval
        time.sleep(interval)

if __name__ == "__main__":
    main_inference()
