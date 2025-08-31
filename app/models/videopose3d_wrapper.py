import os
import sys
import numpy as np
import torch

# Add the project root to the Python path to allow top-level imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the model from the VideoPose3d directory at the root
from VideoPose3d.common.model import TemporalModel


class LiftError(Exception):
    """Custom exception for lifting errors."""
    pass

class VideoPose3DLifter:
    """
    A wrapper class for the VideoPose3D temporal model to lift 2D poses to 3D.
    """
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        if not os.path.exists(checkpoint_path):
            raise LiftError(f"Checkpoint not found: {checkpoint_path}")

        self.model = TemporalModel(num_joints_in=17,
                                   in_features=2,
                                   num_joints_out=17,
                                   filter_widths=[3, 3, 3, 3, 3]
                                   ).to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt.get("model_pos", ckpt)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        self.receptive_field = self.model.receptive_field()
        print(f"VideoPose3D model loaded successfully. Receptive field: {self.receptive_field} frames.")

    def _normalize(self, keypoints_2d: np.ndarray) -> np.ndarray:
        kp = keypoints_2d.copy()
        pelvis = (kp[:, 11, :] + kp[:, 12, :]) / 2.0
        kp -= pelvis[:, np.newaxis, :]
        
        shoulders = (kp[:, 5, :] + kp[:, 6, :]) / 2.0
        torso = np.linalg.norm(shoulders, axis=1)
        
        torso[torso == 0] = 1.0
        
        kp /= torso[:, np.newaxis, np.newaxis]
        return kp

    @torch.no_grad()
    def lift(self, keypoints_2d: np.ndarray) -> np.ndarray:
        if keypoints_2d.ndim != 3 or keypoints_2d.shape[1:] != (17, 2):
            raise LiftError(f"Input must be a numpy array of shape [T, 17, 2], but got {keypoints_2d.shape}")

        num_frames = keypoints_2d.shape[0]
        
        pad = self.receptive_field - 1
        pad_left = pad // 2
        pad_right = pad - pad_left
        
        input_keypoints = np.pad(keypoints_2d, ((pad_left, pad_right), (0, 0), (0, 0)), 'edge')

        kp_norm = self._normalize(input_keypoints)
        x = torch.from_numpy(kp_norm).float().unsqueeze(0).to(self.device)
        out_tensor = self.model(x)

        out_np = out_tensor.squeeze(0).cpu().numpy()
        pelvis_3d = (out_np[:, 11, :] + out_np[:, 12, :]) / 2.0
        out_np -= pelvis_3d[:, np.newaxis, :]
        
        return out_np