from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
import numpy as np

# Create a router object. All endpoints defined with this router
# will be grouped together.
router = APIRouter()

class Pose2DRequest(BaseModel):
    keypoints_2d: list

# Define the endpoint at the path "/predict_3d".
# The full URL will become "/v1/predict_3d" after the prefix is added in main.py.
@router.post("/predict_3d")
async def predict_3d(request: Request, body: Pose2DRequest):
    """
    Receives 2D keypoints and returns a 3D pose prediction.
    """
    # Get the lifter object that was initialized at startup
    lifter = request.app.state.lifter
    
    try:
        kp = np.array(body.keypoints_2d, dtype=np.float32)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid data in keypoints_2d.")
        
    if kp.ndim != 3 or kp.shape[1:] != (17, 2):
        raise HTTPException(status_code=422, detail=f"Incorrect shape {kp.shape}. Expected: [T, 17, 2].")
        
    try:
        # Call the lift method
        kp3d = lifter.lift(kp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during 3D lifting: {repr(e)}")
        
    return {
        "keypoints_3d": kp3d.tolist(),
        "frames": int(kp3d.shape[0])
    }