"""
Kalman filter for tracking bounding boxes over time.

State vector (8-dim):
    [cx, cy, aspect_ratio, height, vcx, vcy, vaspect, vheight]

We track the box as (center_x, center_y, aspect_ratio, height) instead of
(x1, y1, x2, y2) because it behaves more linearly under a constant-velocity
motion model -- this is the same parameterization used in SORT/DeepSORT/
ByteTrack.

This filter answers two questions every frame:
    1. predict(): "where do I expect this box to be NOW, before seeing
       the new detections?" (motion compensation)
    2. update(measurement): "given the box I actually detected, what's
       my corrected best estimate?" (fuses prediction + observation)
"""

import numpy as np


class KalmanBoxTracker:
    """Kalman filter wrapper around a single tracked bounding box."""

    count = 0  # global counter used to hand out unique track IDs

    def __init__(self, bbox_xyxy: np.ndarray):
        """
        Args:
            bbox_xyxy: initial detection box as [x1, y1, x2, y2].
        """
        # ---- state transition matrix (constant velocity model) ----
        # next_position = position + velocity * dt   (dt = 1 frame)
        self.ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * self.ndim, 2 * self.ndim)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = dt

        # ---- observation matrix: we only directly observe [cx,cy,a,h] ----
        self._update_mat = np.eye(self.ndim, 2 * self.ndim)

        # noise: how much we trust the model vs. the measurement.
        # Tuned empirically (standard values used across SORT-family trackers).
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

        cx, cy, a, h = self._xyxy_to_z(bbox_xyxy)
        self.mean = np.array([cx, cy, a, h, 0, 0, 0, 0], dtype=float)

        std = [
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            1e-2,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            1e-5,
            10 * self._std_weight_velocity * h,
        ]
        self.covariance = np.diag(np.square(std))

        KalmanBoxTracker.count += 1
        self.track_id = KalmanBoxTracker.count
        self.time_since_update = 0
        self.hits = 1
        self.age = 0

    @staticmethod
    def _xyxy_to_z(bbox_xyxy: np.ndarray):
        """[x1,y1,x2,y2] -> [cx, cy, aspect_ratio, height]."""
        x1, y1, x2, y2 = bbox_xyxy
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2.0, y1 + h / 2.0
        a = w / max(h, 1e-6)
        return cx, cy, a, h

    def to_xyxy(self) -> np.ndarray:
        """Current state -> [x1, y1, x2, y2]."""
        cx, cy, a, h = self.mean[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def predict(self) -> None:
        """Advance the state one frame using the motion model (no measurement)."""
        h = self.mean[3]
        std_pos = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h,
        ]
        std_vel = [
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5,
            self._std_weight_velocity * h,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        self.mean = self._motion_mat @ self.mean
        self.covariance = (
            self._motion_mat @ self.covariance @ self._motion_mat.T + motion_cov
        )
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox_xyxy: np.ndarray) -> None:
        """Correct the prediction using a new measured bounding box."""
        cx, cy, a, h = self._xyxy_to_z(bbox_xyxy)
        measurement = np.array([cx, cy, a, h])

        std = [
            self._std_weight_position * self.mean[3],
            self._std_weight_position * self.mean[3],
            1e-1,
            self._std_weight_position * self.mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        proj_mean = self._update_mat @ self.mean
        proj_cov = self._update_mat @ self.covariance @ self._update_mat.T + innovation_cov

        kalman_gain = (
            self.covariance @ self._update_mat.T @ np.linalg.inv(proj_cov)
        )
        innovation = measurement - proj_mean

        self.mean = self.mean + kalman_gain @ innovation
        self.covariance = self.covariance - kalman_gain @ self._update_mat @ self.covariance

        self.time_since_update = 0
        self.hits += 1
