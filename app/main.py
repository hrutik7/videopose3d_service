# main.py

import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- START: Bulletproof Import Fix ---
# Ensures that modules can be found regardless of how the app is run.
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up to the project root to find 'models' and 'VideoPose3d'
project_root = current_dir 
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- END: Bulletproof Import Fix ---

# Import your API router and the model wrapper
from .api import router
from .models.videopose3d_wrapper import VideoPose3DLifter

app = FastAPI(title="VideoPose3D Lifter")

# --- START: CORS Configuration ---
# Allows your frontend (e.g., running on localhost:3000) to communicate with this backend.
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
# --- END: CORS Configuration ---

# Include the API endpoints defined in api.py, prefixed with /v1
app.include_router(router, prefix="/v1")

@app.on_event("startup")
async def startup_event():
    """
    This function runs once when the application starts.
    It loads the VideoPose3D model into memory so it's ready for predictions.
    """
    # Define the path to the pre-trained model checkpoint.
    # It looks for an environment variable first, otherwise uses the default path.
    ckpt_path = os.getenv("VPOSE_CHECKPOINT", "./VideoPose3d/checkpoint/pretrained_h36m_cpn.bin")
    
    # Use 'cuda' for GPU if available, otherwise 'cpu'.
    device = os.getenv("VPOSE_DEVICE", "cpu")
    
    # Instantiate the lifter and store it in the app's state for access in API endpoints.
    print(f"Loading model from: {ckpt_path} onto device: {device}")
    app.state.lifter = VideoPose3DLifter(checkpoint_path=ckpt_path, device=device)
    print("Model loaded and application started.")

@app.get("/")
def root():
    """A simple root endpoint to confirm the API is running."""
    return {"message": "VideoPose3D API is running!"}