import os
import sys
import numpy as np
import torch

# This ensures the 'VideoPose3d' module can be found from this file's location.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Initialize the TemporalModel with the architecture for Human3.6M (17 joints).
        self.model = TemporalModel(
            num_joints_in=17,
            in_features=2,      # (x, y)
            num_joints_out=17,
            filter_widths=[3, 3, 3, 3, 3] # This is the standard receptive field setup
        ).to(self.device)

        # Load the pre-trained weights.
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt.get("model_pos", ckpt)
        self.model.load_state_dict(state, strict=False)
        self.model.eval() # Set the model to evaluation mode.

        self.receptive_field = self.model.receptive_field()
        print(f"VideoPose3D model loaded. Receptive field: {self.receptive_field} frames.")

    def _normalize(self, keypoints_2d: np.ndarray) -> np.ndarray:
        """Normalizes the 2D keypoints for the model."""
        kp = keypoints_2d.copy()
        # Center the pose at the pelvis.
        pelvis = (kp[:, 11, :] + kp[:, 12, :]) / 2.0 # COCO format: left_hip=11, right_hip=12
        kp -= pelvis[:, np.newaxis, :]
        
        # Scale the pose based on the torso length.
        shoulders = (kp[:, 5, :] + kp[:, 6, :]) / 2.0 # left_shoulder=5, right_shoulder=6
        torso = np.linalg.norm(shoulders, axis=1)
        
        # Avoid division by zero
        torso[torso == 0] = 1.0
        
        kp /= torso[:, np.newaxis, np.newaxis]
        return kp

    @torch.no_grad()
    def lift(self, keypoints_2d: np.ndarray) -> np.ndarray:
        """
        Takes a NumPy array of 2D keypoints [T, 17, 2] and returns 3D keypoints [T, 17, 3].
        """
        if keypoints_2d.ndim != 3 or keypoints_2d.shape[1:] != (17, 2):
            raise LiftError(f"Input must be a numpy array of shape [T, 17, 2], but got {keypoints_2d.shape}")

        num_frames = keypoints_2d.shape[0]
        
        # Pad the sequence to account for the model's receptive field.
        # This ensures the model has enough context for the first and last frames.
        pad = (self.receptive_field - 1) // 2
        input_keypoints = np.pad(keypoints_2d, ((pad, pad), (0, 0), (0, 0)), 'edge')

        # Normalize, convert to tensor, and run through the model.
        kp_norm = self._normalize(input_keypoints)
        x = torch.from_numpy(kp_norm).float().unsqueeze(0).to(self.device)
        out_tensor = self.model(x)

        # Convert the output back to a NumPy array and post-process.
        out_np = out_tensor.squeeze(0).cpu().numpy()
        # Recenter the 3D pose at the pelvis for consistency.
        pelvis_3d = (out_np[:, 11, :] + out_np[:, 12, :]) / 2.0
        out_np -= pelvis_3d[:, np.newaxis, :]
        
        return out_np