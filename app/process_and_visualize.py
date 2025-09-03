# File: process_and_visualize.py

import requests
import json
import numpy as np

# --- Configuration ---
# API to get the raw 2D exercise data from
SOURCE_API_URL = "https://physisyncv100-production.up.railway.app/exercise/get-exercise-patient/ba6843c4-208f-435f-b993-6d2863f551e1"

# Your local API that performs the 2D-to-3D conversion
PREDICTION_API_URL = "http://localhost:8000/v1/predict_3d"

# The order of keypoints expected by the 3D model
TARGET_KEYPOINT_ORDER = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

def parse_exercisedata(exercisedata_str: str) -> (np.ndarray, list):
    """
    Parses a single 'exercisedata' string, which contains escaped JSON.
    
    Returns a NumPy array of 2D keypoints and a list of timestamps.
    Returns (None, None) if parsing fails or data is empty.
    """
    if not exercisedata_str or exercisedata_str == '""':
        return None, None

    try:
        # Step 1: Clean the string by removing outer quotes and un-escaping inner quotes
        cleaned_str = exercisedata_str[1:-1].replace('\\"', '"')
        frames_data = json.loads(cleaned_str)
    except (json.JSONDecodeError, IndexError) as e:
        print(f"    - Warning: Could not parse JSON for a session. Error: {e}")
        return None, None

    all_frames_ordered = []
    timestamps = []

    # Step 2: Iterate through each frame in the data
    for frame in frames_data:
        timestamps.append(frame.get('timestamp', None))
        
        keypoints_map = {kp['name']: (kp['x'], kp['y']) for kp in frame.get('keypoints', [])}
        
        current_frame_ordered = []
        for keypoint_name in TARGET_KEYPOINT_ORDER:
            # Default to (0,0) if a keypoint is missing
            coords = keypoints_map.get(keypoint_name, (0.0, 0.0))
            current_frame_ordered.append(list(coords))
        
        all_frames_ordered.append(current_frame_ordered)
    
    if not all_frames_ordered:
        return None, None
        
    keypoints_array = np.array(all_frames_ordered, dtype=np.float32)
    return keypoints_array, timestamps

def main():
    """
    Main function to orchestrate the entire data pipeline.
    """
    # --- STAGE 1: FETCH RAW DATA FROM SOURCE API ---
    print(f"Fetching 2D exercise data from:\n{SOURCE_API_URL}\n")
    try:
        response = requests.get(SOURCE_API_URL, timeout=30)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        source_data = response.json()
        print(f"Successfully fetched {len(source_data)} exercise session(s).\n")
    except requests.exceptions.RequestException as e:
        print(f"Fatal Error: Could not fetch data from the source API. {e}")
        return # Exit the script

    if not source_data:
        print("No exercise sessions found for this patient.")
        return

    # This list will hold the final 3D data for ALL sessions
    all_sessions_3d_output = []

    # --- STAGE 2: PROCESS EACH SESSION INDIVIDUALLY ---
    for index, session in enumerate(source_data):
        session_id = session.get("id", f"UnknownID-{index}")
        print(f"--- Processing Session {index + 1}/{len(source_data)} (ID: {session_id}) ---")

        # Step 2a: Parse the string data for the current session
        exercisedata_str = session.get("exercisedata")
        keypoints_2d_array, frame_timestamps = parse_exercisedata(exercisedata_str)

        if keypoints_2d_array is None or not frame_timestamps:
            print("    - Skipping session due to empty or invalid exercise data.\n")
            continue
            
        print(f"    - Parsed {len(frame_timestamps)} frames of 2D keypoints.")

        # Step 2b: Send the parsed data to the 3D prediction API
        payload = {
            "keypoints_2d": keypoints_2d_array.tolist(),
            "timestamps": frame_timestamps
        }

        print(f"    - Sending {len(frame_timestamps)} frames to local 3D prediction API...")
        try:
            prediction_response = requests.post(PREDICTION_API_URL, json=payload, timeout=60)
            prediction_response.raise_for_status()
            
            # The 3D API should return a list of {'timestamp': ..., 'keypoints_3d': ...}
            session_3d_data = prediction_response.json()
            
            # Add original session info to the final output
            all_sessions_3d_output.append({
                "original_session_id": session_id,
                "patientId": session.get("patientId"),
                "exerciseId": session.get("exerciseId"),
                "processed_frames": session_3d_data
            })
            print(f"    - Successfully received 3D data for {len(session_3d_data)} frames.\n")

        except requests.exceptions.RequestException as e:
            print(f"    - Error: Failed to get 3D prediction for session {session_id}. {e}\n")

    # --- STAGE 3: SAVE THE FINAL COMBINED DATA ---
    if not all_sessions_3d_output:
        print("Processing complete, but no valid 3D data was generated.")
        return
        
    output_filename = 'visualization_output.json'
    with open(output_filename, 'w') as f:
        # indent=2 makes the JSON file readable
        # --- FIX WAS HERE ---
        json.dump(all_sessions_3d_output, f, indent=2)

    print("=================================================================")
    print("SUCCESS: All sessions processed.")
    print(f"Final 3D visualization data has been saved to: {output_filename}")
    print("You can now use this file in your React front-end.")
    print("=================================================================")


if __name__ == "__main__":
    main()