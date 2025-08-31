from fastapi import FastAPI
from .api import router
from .models.videopose3d_wrapper import VideoPose3DLifter
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
# --- START: Bulletproof Import Fix ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- END: Bulletproof Import Fix ---


# Import the CORS middleware


app = FastAPI(title="VideoPose3D Lifter")

# --- START: CORS Configuration ---
# Define the list of origins that are allowed to make requests.
# You should restrict this to your actual frontend URL in production.
origins = [
    "http://localhost",
    "http://localhost:3000", # The default for create-react-app
    "http://localhost:5174", # The default for create-react-app
    "http://localhost:5173", # The default for create-react-app
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)
# --- END: CORS Configuration ---

app.include_router(router, prefix="/v1")

@app.on_event("startup")
async def startup_event():
    # default checkpoint path (you can override via env)
    ckpt = os.getenv("VPOSE_CHECKPOINT", "./VideoPose3d/checkpoint/pretrained_h36m_cpn.bin")
    device = os.getenv("VPOSE_DEVICE", "cpu")
    # instantiate lifter once
    app.state.lifter = VideoPose3DLifter(checkpoint_path=ckpt, device=device)

@app.get("/")
def root():
    return {"message": "PoseNet API is running!"}
