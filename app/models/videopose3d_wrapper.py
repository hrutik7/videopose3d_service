import os, sys, numpy as np, torch
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.insert(0, project_root)
from VideoPose3d.common.model import TemporalModel
class LiftError(Exception): pass
class VideoPose3DLifter:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        if not os.path.exists(checkpoint_path): raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        self.model = TemporalModel(num_joints_in=17, in_features=2, num_joints_out=17, filter_widths=[3, 3, 3, 3, 3]).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt.get("model_pos", ckpt)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.receptive_field = self.model.receptive_field()
        print(f"VideoPose3D model loaded. Receptive field: {self.receptive_field} frames.")
    def _normalize(self, keypoints_2d: np.ndarray) -> np.ndarray:
        kp = keypoints_2d.copy()
        pelvis = (kp[:, 11, :] + kp[:, 12, :]) / 2.0; kp -= pelvis[:, np.newaxis, :]
        shoulders = (kp[:, 5, :] + kp[:, 6, :]) / 2.0; torso = np.linalg.norm(shoulders, axis=1)
        torso[torso == 0] = 1.0; kp /= torso[:, np.newaxis, np.newaxis]
        return kp
    @torch.no_grad()
    def lift(self, keypoints_2d: np.ndarray) -> np.ndarray:
        if keypoints_2d.ndim != 3 or keypoints_2d.shape[1:] != (17, 2): raise LiftError(f"Input shape must be [T, 17, 2], but got {keypoints_2d.shape}")
        pad = (self.receptive_field - 1) // 2
        input_keypoints = np.pad(keypoints_2d, ((pad, pad), (0, 0), (0, 0)), 'edge')
        kp_norm = self._normalize(input_keypoints); x = torch.from_numpy(kp_norm).float().unsqueeze(0).to(self.device)
        out_tensor = self.model(x); out_np = out_tensor.squeeze(0).cpu().numpy()
        pelvis_3d = (out_np[:, 11, :] + out_np[:, 12, :]) / 2.0; out_np -= pelvis_3d[:, np.newaxis, :]
        transformed_out_np = np.zeros_like(out_np)
        transformed_out_np[..., 0] = out_np[..., 0]; transformed_out_np[..., 1] = -out_np[..., 2]; transformed_out_np[..., 2] = -out_np[..., 1]
        return transformed_out_np