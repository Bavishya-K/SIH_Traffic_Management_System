import streamlit as st
import time
import pandas as pd
import numpy as np
from collections import deque
from traffic_count_detection_1 import get_vehicle_counts_at_second
from emergency_detection import detect_emergency_at_second

# Initialize State Variables
lane_names = ["South", "North", "East", "West"]
waiting_times = {lane: 0 for lane in lane_names}
previous_lane = None  # Track previously activated lane
consecutive_activations = {lane: 0 for lane in lane_names}  # Track consecutive activations for each lane
base_green_duration = 20  # Base green signal duration

# Initialize Dashboard
st.title("Real-Time Traffic Signal Simulation")
status_placeholder = st.empty()
chart_placeholder1 = st.empty()
chart_placeholder2 = st.empty()
table_placeholder = st.empty()

# Traffic Signal Logic with Dynamic Green Time
def optimize_signals(in_count, out_count, emergency, waiting_times, neighbor_state, previous_lane, alpha=0.5, beta=0.7):
    global consecutive_activations
    base_green_duration = 20  # Minimum green signal duration
    max_extra_time = 30 # Maximum extra time for dynamic adjustment

    # Emergency Vehicle Handling
    if emergency in lane_names:
        return {
            "green": emergency, 
            "red": [d for d in lane_names if d != emergency], 
            "duration": base_green_duration
        }

    # Calculate Priority for Each Lane
    priorities = []
    for i, lane in enumerate(lane_names):
        in_count_value = in_count[i] if in_count[i] is not None else 0
        penalty_factor = 0.5  # Reduce the priority by 50% for consecutive activations
        max_consecutive = 2  # Limit consecutive activations to 3 times

        
        # Apply penalty if lane has been activated consecutively more than the limit
        if consecutive_activations[lane] >= max_consecutive:
            penalty = penalty_factor
        else:
            penalty = 1  # No penalty

        waiting_time_value = waiting_times.get(lane, 0)
        neighbor_state_value = neighbor_state[i] if neighbor_state[i] is not None else 0
        priority_score = in_count_value + alpha * waiting_time_value + beta * neighbor_state_value * penalty
        priorities.append(priority_score)

    # Select the Lane with Maximum Priority
    max_priority = max(priorities)
    selected_lane = lane_names[priorities.index(max_priority)]

    # Determine Dynamic Green Time Based on Traffic
    vehicle_count = in_count[lane_names.index(selected_lane)]
    extra_time = min(vehicle_count * 2, max_extra_time)  # Add 2s per vehicle, cap at max_extra_time
    green_duration = base_green_duration + extra_time

    # Update Waiting Times for Other Lanes
    for lane in lane_names:
        if lane != selected_lane:
            waiting_times[lane] += green_duration
        else:
            waiting_times[lane] = 0

    # Update Consecutive Activations
    # Update Consecutive Activations
    if selected_lane == previous_lane:
        consecutive_activations[selected_lane] += 1
    else:
        consecutive_activations[selected_lane] = 1  # Reset count for the newly selected lane, start counting from 1
    # Reset all lanes

    return {
        "green": selected_lane, 
        "red": [d for d in lane_names if d != selected_lane], 
        "duration": green_duration
    }

# Main Control Loop
def traffic_signal_control():
    target_second = 0
    global previous_lane

    # Manually set neighbor state for simplicity
    neighbor_state = [5, 7, 3, 4]

    while True:
        in_counts, out_counts, emergency = [], [], None
        
        # Collect traffic data
        for i, lane in enumerate(lane_names):
            in_count, out_count = get_vehicle_counts_at_second(f"../videos/India/lane_{i+1}.mp4", target_second)
            in_counts.append(in_count if in_count is not None else 0)
            out_counts.append(out_count if out_count is not None else 0)
            emergency = detect_emergency_at_second(f"../videos/cut/lane_{i+1}.mp4", target_second)

        # Optimize signals dynamically
        signal_states = optimize_signals(in_counts, out_counts, emergency, waiting_times, neighbor_state, previous_lane)
        green_lane = signal_states['green']
        green_duration = signal_states['duration']

        # Display Dashboard
        status_placeholder.markdown(f"### 🟢 Green: {green_lane} ({green_duration}s) | 🔴 Red: {', '.join(signal_states['red'])}")

        # Update traffic data display
        traffic_df = pd.DataFrame({
            "Lane": lane_names,
            "Vehicles Waiting": in_counts,
            "Waiting Time (s)": [waiting_times[lane] for lane in lane_names],
            "Neighbor Traffic": neighbor_state,
            "Emergency": [emergency if emergency == lane else "No" for lane in lane_names]
        })

        table_placeholder.table(traffic_df)
        chart_placeholder1.bar_chart(traffic_df.set_index("Lane")["Vehicles Waiting"], use_container_width=True)
        chart_placeholder2.bar_chart(traffic_df.set_index("Lane")["Waiting Time (s)"], use_container_width=True)

        previous_lane = green_lane
        target_second += green_duration
        time.sleep(green_duration)

# Start the Simulation
if __name__ == "__main__":
    traffic_signal_control()