import json
import numpy as np
import requests # To make the HTTP request

def parse_and_prepare_data(raw_data: dict) -> np.ndarray:
    """
    Parses the specific 'exercisedata' format, re-orders keypoints,
    and returns a NumPy array in the shape [T, 17, 2].
    """
    # The order of keypoints expected by the VideoPose3D model (COCO/Human3.6M format)
    # This specific order is crucial.
    TARGET_KEYPOINT_ORDER = [
        "nose",             # 0
        "left_eye",         # 1
        "right_eye",        # 2
        "left_ear",         # 3
        "right_ear",        # 4
        "left_shoulder",    # 5
        "right_shoulder",   # 6
        "left_elbow",       # 7
        "right_elbow",      # 8
        "left_wrist",       # 9
        "right_wrist",      # 10
        "left_hip",         # 11
        "right_hip",        # 12
        "left_knee",        # 13
        "right_knee",       # 14
        "left_ankle",       # 15
        "right_ankle"       # 16
    ]

    # Step 1: Parse the string-escaped JSON
    exercisedata_str = raw_data["exercisedata"]
    # The [1:-1] removes the outer quotes from the string before parsing
    frames_data = json.loads(exercisedata_str[1:-1])

    all_frames_ordered = []

    # Step 2: Iterate through each frame in the data
    for frame in frames_data:
        # Create a dictionary for quick lookup of keypoints by name
        keypoints_map = {kp['name']: (kp['x'], kp['y']) for kp in frame['keypoints']}
        
        current_frame_ordered = []
        # Step 3: Re-order the keypoints according to TARGET_KEYPOINT_ORDER
        for keypoint_name in TARGET_KEYPOINT_ORDER:
            # Get the (x, y) tuple, defaulting to (0, 0) if a keypoint is missing
            coords = keypoints_map.get(keypoint_name, (0.0, 0.0))
            current_frame_ordered.append(list(coords))
        
        all_frames_ordered.append(current_frame_ordered)
        
    # Step 4: Convert the list of frames to a NumPy array
    return np.array(all_frames_ordered, dtype=np.float32)


# Your original raw data
# NOTE: For a real test, you should have multiple frames. I've duplicated your frame
# data 243 times to simulate a sequence, as the model's receptive field is 243 frames.
raw_data_string = "\"[{\\\"timestamp\\\":1755690525783,\\\"keypoints\\\":[{\\\"y\\\":132.47,\\\"x\\\":609.74,\\\"score\\\":0.64,\\\"name\\\":\\\"nose\\\"},{\\\"y\\\":123.57,\\\"x\\\":620.83,\\\"score\\\":0.48,\\\"name\\\":\\\"left_eye\\\"},{\\\"y\\\":123.32,\\\"x\\\":597.63,\\\"score\\\":0.61,\\\"name\\\":\\\"right_eye\\\"},{\\\"y\\\":136.48,\\\"x\\\":638.37,\\\"score\\\":0.62,\\\"name\\\":\\\"left_ear\\\"},{\\\"y\\\":136.93,\\\"x\\\":580.98,\\\"score\\\":0.57,\\\"name\\\":\\\"right_ear\\\"},{\\\"y\\\":194.16,\\\"x\\\":672.22,\\\"score\\\":0.80,\\\"name\\\":\\\"left_shoulder\\\"},{\\\"y\\\":199.73,\\\"x\\\":543.80,\\\"score\\\":0.71,\\\"name\\\":\\\"right_shoulder\\\"},{\\\"y\\\":194.29,\\\"x\\\":758.29,\\\"score\\\":0.67,\\\"name\\\":\\\"left_elbow\\\"},{\\\"y\\\":211.18,\\\"x\\\":470.85,\\\"score\\\":0.71,\\\"name\\\":\\\"right_elbow\\\"},{\\\"y\\\":132.66,\\\"x\\\":755.61,\\\"score\\\":0.26,\\\"name\\\":\\\"left_wrist\\\"},{\\\"y\\\":154.04,\\\"x\\\":390.63,\\\"score\\\":0.42,\\\"name\\\":\\\"right_wrist\\\"},{\\\"y\\\":356.10,\\\"x\\\":646.96,\\\"score\\\":0.78,\\\"name\\\":\\\"left_hip\\\"},{\\\"y\\\":357.02,\\\"x\\\":578.38,\\\"score\\\":0.80,\\\"name\\\":\\\"right_hip\\\"},{\\\"y\\\":519.36,\\\"x\\\":652.54,\\\"score\\\":0.86,\\\"name\\\":\\\"left_knee\\\"},{\\\"y\\\":518.12,\\\"x\\\":577.53,\\\"score\\\":0.87,\\\"name\\\":\\\"right_knee\\\"},{\\\"y\\\":668.51,\\\"x\\\":647.51,\\\"score\\\":0.85,\\\"name\\\":\\\"left_ankle\\\"},{\\\"y\\\":665.93,\\\"x\\\":579.18,\\\"score\\\":0.78,\\\"name\\\":\\\"right_ankle\\\"}]}]\""
duplicated_frames = ",".join(raw_data_string[3:-3] for _ in range(243))
final_raw_string = f"\"[{duplicated_frames}]\""


raw_data_full = {
    "exercisedata": final_raw_string
}


# --- Main execution ---
if __name__ == "__main__":
    print("Parsing and preparing data...")
    keypoints_2d_array = parse_and_prepare_data(raw_data_full)
    
    print(f"Data prepared successfully. Shape: {keypoints_2d_array.shape}")
    
    # Define the API endpoint
    url = "http://localhost:8000/v1/predict_3d"
    
    # Create the JSON payload for the request
    payload = {
        "keypoints_2d": keypoints_2d_array.tolist()
    }
    
    print("\nSending request to API...")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        result = response.json()
        print("\nAPI Response Received Successfully!")
        print(f"Frames processed: {result['frames_processed']}")
        
        # Print the 3D coordinates of the nose for the first frame as an example
        first_frame_3d = result['keypoints_3d'][0]
        nose_3d = first_frame_3d[0] # Index 0 is 'nose'
        print(f"Example - 3D coordinates for 'nose' in the first frame: {nose_3d}")
        
    except requests.exceptions.RequestException as e:
        print(f"\nAn error occurred while calling the API: {e}")

# --- Add these two lines to the end of your script ---
with open('payload.json', 'w') as f:
    json.dump(payload, f)
print("\nPayload for curl has been saved to payload.json")