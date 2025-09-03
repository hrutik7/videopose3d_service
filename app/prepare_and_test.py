# prepare_and_test.py

import json
import numpy as np
import requests # To make the HTTP request

def parse_and_prepare_data(raw_data: dict) -> (np.ndarray, list):
    """
    Parses the 'exercisedata' format, re-orders keypoints,
    and returns a NumPy array of 2D keypoints and a list of timestamps.
    """
    TARGET_KEYPOINT_ORDER = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    exercisedata_str = raw_data["exercisedata"]
    cleaned_exercisedata_str = exercisedata_str[1:-1].replace('\\"', '"')
    frames_data = json.loads(cleaned_exercisedata_str)

    all_frames_ordered = []
    timestamps = []

    for frame in frames_data:
        timestamps.append(frame.get('timestamp', None))
        keypoints_map = {kp['name']: (kp['x'], kp['y']) for kp in frame['keypoints']}
        
        current_frame_ordered = []
        for keypoint_name in TARGET_KEYPOINT_ORDER:
            coords = keypoints_map.get(keypoint_name, (0.0, 0.0))
            current_frame_ordered.append(list(coords))
        
        all_frames_ordered.append(current_frame_ordered)
        
    keypoints_array = np.array(all_frames_ordered, dtype=np.float32)
    return keypoints_array, timestamps


# Your original raw data
raw_data_string = "\"[{\\\"timestamp\\\":1755690525783,\\\"keypoints\\\":[{\\\"y\\\":132.47,\\\"x\\\":609.74,\\\"score\\\":0.64,\\\"name\\\":\\\"nose\\\"},{\\\"y\\\":123.57,\\\"x\\\":620.83,\\\"score\\\":0.48,\\\"name\\\":\\\"left_eye\\\"},{\\\"y\\\":123.32,\\\"x\\\":597.63,\\\"score\\\":0.61,\\\"name\\\":\\\"right_eye\\\"},{\\\"y\\\":136.48,\\\"x\\\":638.37,\\\"score\\\":0.62,\\\"name\\\":\\\"left_ear\\\"},{\\\"y\\\":136.93,\\\"x\\\":580.98,\\\"score\\\":0.57,\\\"name\\\":\\\"right_ear\\\"},{\\\"y\\\":194.16,\\\"x\\\":672.22,\\\"score\\\":0.80,\\\"name\\\":\\\"left_shoulder\\\"},{\\\"y\\\":199.73,\\\"x\\\":543.80,\\\"score\\\":0.71,\\\"name\\\":\\\"right_shoulder\\\"},{\\\"y\\\":194.29,\\\"x\\\":758.29,\\\"score\\\":0.67,\\\"name\\\":\\\"left_elbow\\\"},{\\\"y\\\":211.18,\\\"x\\\":470.85,\\\"score\\\":0.71,\\\"name\\\":\\\"right_elbow\\\"},{\\\"y\\\":132.66,\\\"x\\\":755.61,\\\"score\\\":0.26,\\\"name\\\":\\\"left_wrist\\\"},{\\\"y\\\":154.04,\\\"x\\\":390.63,\\\"score\\\":0.42,\\\"name\\\":\\\"right_wrist\\\"},{\\\"y\\\":356.10,\\\"x\\\":646.96,\\\"score\\\":0.78,\\\"name\\\":\\\"left_hip\\\"},{\\\"y\\\":357.02,\\\"x\\\":578.38,\\\"score\\\":0.80,\\\"name\\\":\\\"right_hip\\\"},{\\\"y\\\":519.36,\\\"x\\\":652.54,\\\"score\\\":0.86,\\\"name\\\":\\\"left_knee\\\"},{\\\"y\\\":518.12,\\\"x\\\":577.53,\\\"score\\\":0.87,\\\"name\\\":\\\"right_knee\\\"},{\\\"y\\\":668.51,\\\"x\\\":647.51,\\\"score\\\":0.85,\\\"name\\\":\\\"left_ankle\\\"},{\\\"y\\\":665.93,\\\"x\\\":579.18,\\\"score\\\":0.78,\\\"name\\\":\\\"right_ankle\\\"}]}]\""

duplicated_frames = ",".join(raw_data_string[2:-2] for _ in range(243))
final_raw_string = f"\"[{duplicated_frames}]\""

raw_data_full = {
    "exercisedata": final_raw_string
}

# --- Main execution ---
if __name__ == "__main__":
    print("Parsing and preparing data...")
    keypoints_2d_array, frame_timestamps = parse_and_prepare_data(raw_data_full)
    
    print(f"Data prepared successfully. Shape: {keypoints_2d_array.shape}")
    print(f"Timestamps collected: {len(frame_timestamps)}")
    
    url = "http://localhost:8000/v1/predict_3d"
    
    # --- MODIFIED: Add timestamps to the payload ---
    payload = {
        "keypoints_2d": keypoints_2d_array.tolist(),
        "timestamps": frame_timestamps # Send timestamps to the API
    }
    
    print("\nSending request to API...")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        # --- MODIFIED: The result is now the final structured data ---
        final_output = response.json()
        print("\nAPI Response Received Successfully!")
        print(f"Frames processed: {len(final_output)}")

        # Example: Print the data for the first frame from the API response
        if final_output:
            first_frame_data = final_output[0]
            print("\nExample - Data for the first frame:")
            print(f"  Timestamp: {first_frame_data['timestamp']}")
            print(f"  3D Coordinates for 'nose': {first_frame_data['keypoints_3d'][0]}") # Index 0 is 'nose'
        
    except requests.exceptions.RequestException as e:
        print(f"\nAn error occurred while calling the API: {e}")

    # Save the payload that was sent for debugging
    with open('payload.json', 'w') as f:
        json.dump(payload, f)
    print("\nPayload sent to API has been saved to payload.json")