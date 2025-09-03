# File: api.py

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import List, Optional
import requests  # <-- Add this import to make external API calls
import json      # <-- Add this import for JSON parsing

# Create a router object.
router = APIRouter()

# The order of keypoints expected by the 3D model
TARGET_KEYPOINT_ORDER = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


# ==============================================================================
# HELPER FUNCTION (from your script)
# ==============================================================================
def parse_exercisedata(exercisedata_str: str) -> (np.ndarray, list):
    """
    Parses a single 'exercisedata' string, which contains escaped JSON.
    Returns a NumPy array of 2D keypoints and a list of timestamps.
    """
    if not exercisedata_str or exercisedata_str == '""':
        return None, None
    try:
        cleaned_str = exercisedata_str[1:-1].replace('\\"', '"')
        frames_data = json.loads(cleaned_str)
    except (json.JSONDecodeError, IndexError):
        return None, None

    all_frames_ordered, timestamps = [], []
    for frame in frames_data:
        timestamps.append(frame.get('timestamp', None))
        keypoints_map = {kp['name']: (kp['x'], kp['y']) for kp in frame.get('keypoints', [])}
        current_frame_ordered = [list(keypoints_map.get(name, (0.0, 0.0))) for name in TARGET_KEYPOINT_ORDER]
        all_frames_ordered.append(current_frame_ordered)
    
    if not all_frames_ordered:
        return None, None
        
    return np.array(all_frames_ordered, dtype=np.float32), timestamps


# ==============================================================================
# NEW: FRONTEND-FACING "ORCHESTRATOR" ENDPOINT
# ==============================================================================
@router.get("/process-patient-sessions/{patient_id}")
async def process_patient_sessions(patient_id: str, request: Request):
    """
    This is the main endpoint for the frontend.
    It fetches patient data, processes it, performs 3D lifting, and returns
    the final structured data ready for visualization.
    """
    source_api_url = f"https://physisyncv100-production.up.railway.app/exercise/get-exercise-patient/{patient_id}"
    
    # --- STAGE 1: FETCH RAW DATA FROM SOURCE API ---
    try:
        response = requests.get(source_api_url, timeout=30)
        response.raise_for_status()
        source_data = response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Could not fetch data from source API: {e}")

    if not source_data:
        return [] # Return an empty list if no sessions are found

    lifter = request.app.state.lifter
    all_sessions_3d_output = []

    # --- STAGE 2: PROCESS EACH SESSION ---
    for session in source_data:
        keypoints_2d_array, frame_timestamps = parse_exercisedata(session.get("exercisedata"))

        if keypoints_2d_array is None or not frame_timestamps:
            continue # Skip empty or invalid sessions
        
        # --- STAGE 3: PERFORM 3D LIFTING (EFFICIENTLY) ---
        # Directly call the lifter object, no need for an internal HTTP request
        try:
            keypoints_3d_np = lifter.lift(keypoints_2d_array)
        except Exception as e:
            # Log this error on the server but continue processing other sessions
            print(f"Error lifting session {session.get('id')}: {e}")
            continue
            
        # Combine results with timestamps
        processed_frames = [
            {"timestamp": ts, "keypoints_3d": kps.tolist()}
            for ts, kps in zip(frame_timestamps, keypoints_3d_np)
        ]
        
        # Add all session info to the final output
        all_sessions_3d_output.append({
            "original_session_id": session.get("id"),
            "patientId": session.get("patientId"),
            "exerciseId": session.get("exerciseId"),
            "processed_frames": processed_frames
        })

    return all_sessions_3d_output


# ==============================================================================
# ORIGINAL "WORKER" ENDPOINT (still useful for other purposes/testing)
# ==============================================================================
class Pose2DRequest(BaseModel):
    keypoints_2d: list
    timestamps: Optional[List[int]] = None

@router.post("/predict_3d")
async def predict_3d(request: Request, body: Pose2DRequest):
    """Receives pre-processed 2D keypoints and returns 3D pose prediction."""
    lifter = request.app.state.lifter
    try:
        keypoints_2d_np = np.array(body.keypoints_2d, dtype=np.float32)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid data format in keypoints_2d.")
    
    # ... (rest of the original predict_3d endpoint logic)
    # This endpoint is no longer called by the frontend directly for this workflow,
    # but it's good to keep it.
    
    if body.timestamps and len(body.timestamps) != keypoints_2d_np.shape[0]:
        raise HTTPException(status_code=422, detail="Timestamp count mismatch.")

    try:
        keypoints_3d_np = lifter.lift(keypoints_2d_np)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during 3D lifting: {repr(e)}")
        
    response_data = [
        {"timestamp": body.timestamps[i] if body.timestamps else None, "keypoints_3d": frame.tolist()}
        for i, frame in enumerate(keypoints_3d_np)
    ]
    return response_data