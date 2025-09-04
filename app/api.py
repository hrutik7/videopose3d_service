from fastapi import APIRouter, Request, HTTPException
import numpy as np
import requests, json, logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()
def parse_exercisedata(exercisedata_str: str) -> (np.ndarray, list):
    if not exercisedata_str or exercisedata_str == '""': return None, None
    try:
        cleaned_str = exercisedata_str[1:-1].replace('\\"', '"')
        frames_data = json.loads(cleaned_str)
    except (json.JSONDecodeError, IndexError): return None, None
    all_frames, timestamps = [], []
    for frame in frames_data:
        timestamps.append(frame.get('timestamp', None))
        keypoints_for_frame = [[kp['x'], kp['y']] for kp in frame.get('keypoints', [])]
        if len(keypoints_for_frame) != 17: continue
        all_frames.append(keypoints_for_frame)
    if not all_frames: return None, None
    return np.array(all_frames, dtype=np.float32), timestamps
@router.get("/process-patient-sessions/{patient_id}")
async def process_patient_sessions(patient_id: str, request: Request):
    source_api_url = f"https://physisyncv100-production.up.railway.app/exercise/get-exercise-patient/{patient_id}"
    try:
        response = requests.get(source_api_url, timeout=30)
        response.raise_for_status()
        source_data = response.json()
    except requests.exceptions.RequestException as e: raise HTTPException(status_code=503, detail=f"Could not fetch data from source API: {e}")
    if not source_data: return []
    lifter = request.app.state.lifter
    all_sessions_3d_output = []
    for session in source_data:
        keypoints_2d_array, frame_timestamps = parse_exercisedata(session.get("exercisedata"))
        if keypoints_2d_array is None or not frame_timestamps or keypoints_2d_array.shape[0] == 0: continue
        try:
            keypoints_3d_np = lifter.lift(keypoints_2d_array)
        except Exception as e:
            print(f"Error lifting session {session.get('id')}: {e}")
            continue
        processed_frames = [{"timestamp": ts, "keypoints_3d": kps.tolist()} for ts, kps in zip(frame_timestamps, keypoints_3d_np)]
        all_sessions_3d_output.append({"original_session_id": session.get("id"), "patientId": session.get("patientId"), "exerciseId": session.get("exerciseId"), "processed_frames": processed_frames})
    return all_sessions_3d_output