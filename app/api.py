# api.py

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
import numpy as np

# Create a router object.
router = APIRouter()

class Pose2DRequest(BaseModel):
    """Defines the expected structure of the incoming JSON request body."""
    keypoints_2d: list

@router.post("/predict_3d")
async def predict_3d(request: Request, body: Pose2DRequest):
    """
    Receives a sequence of 2D keypoints and returns a 3D pose prediction.
    The input shape is expected to be [T, 17, 2], where T is the number of frames.
    """
    # Get the lifter object that was initialized at startup.
    lifter = request.app.state.lifter
    
    try:
        # Convert the input list to a NumPy array for processing.
        keypoints_2d_np = np.array(body.keypoints_2d, dtype=np.float32)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid data format in keypoints_2d.")
        
    # Validate the shape of the input array.
    if keypoints_2d_np.ndim != 3 or keypoints_2d_np.shape[1:] != (17, 2):
        raise HTTPException(
            status_code=422, 
            detail=f"Incorrect shape {keypoints_2d_np.shape}. Expected: [T, 17, 2]."
        )
        
    try:
        # Perform the 2D-to-3D lifting.
        keypoints_3d_np = lifter.lift(keypoints_2d_np)
    except Exception as e:
        # Catch any errors during the model inference.
        raise HTTPException(status_code=500, detail=f"Error during 3D lifting: {repr(e)}")
        
    # Return the 3D keypoints as a list in the JSON response.
    return {
        "keypoints_3d": keypoints_3d_np.tolist(),
        "frames_processed": int(keypoints_3d_np.shape[0])
    }
