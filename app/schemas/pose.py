from pydantic import BaseModel
from typing import List

class Pose2D(BaseModel):
    keypoints_2d: List[List[List[float]]]
