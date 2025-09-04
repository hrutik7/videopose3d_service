import os, sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir 
if project_root not in sys.path: sys.path.insert(0, project_root)
from .api import router
from .models.videopose3d_wrapper import VideoPose3DLifter
app = FastAPI(title="VideoPose3D Lifter")
origins = ["http://localhost", "http://localhost:3000", "http://localhost:5173", "http://localhost:5174"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(router, prefix="/v1")
@app.on_event("startup")
async def startup_event():
    ckpt_path = os.getenv("VPOSE_CHECKPOINT", "./VideoPose3d/checkpoint/pretrained_h36m_cpn.bin")
    device = os.getenv("VPOSE_DEVICE", "cpu")
    print(f"Loading model from: {ckpt_path} onto device: {device}")
    app.state.lifter = VideoPose3DLifter(checkpoint_path=ckpt_path, device=device)
    print("Model loaded and application started.")
@app.get("/")
def root():
    return {"message": "VideoPose3D API is running!"}