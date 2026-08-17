"""
ByteTrack core.

The key idea of ByteTrack (Zhang et al., 2022) vs. plain SORT:
    SORT only tries to match HIGH-confidence detections to tracks. Any
    detection below the confidence threshold is thrown away entirely.
    This loses real objects that are just temporarily occluded or blurry
    (low-confidence but still real), causing avoidable ID switches.

    ByteTrack instead uses ALL detections, in two association passes:
        Pass 1: match HIGH-confidence detections to tracks (as usual).
        Pass 2: take tracks that are STILL unmatched after pass 1, and
                 try to match them against LOW-confidence detections.
                 This recovers occluded/blurry objects instead of losing
                 their track and creating a brand-new ID later.

Track lifecycle:
    New       -> just created from an unmatched high-conf detection
    Tracked   -> currently being successfully matched each frame
    Lost      -> unmatched this frame, but kept alive for a grace period
                 (in case it reappears, e.g. after occlusion)
    Removed   -> lost for too long -> permanently dropped
"""

from enum import Enum
from typing import List

import numpy as np

from kalman_filter import KalmanBoxTracker
from matching import linear_assignment_iou


class TrackState(Enum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


class STrack:
    """A single tracked object: wraps a Kalman filter + bookkeeping."""

    def __init__(self, bbox_xyxy: np.ndarray, score: float):
        self.kf = KalmanBoxTracker(bbox_xyxy)
        self.track_id = self.kf.track_id
        self.score = score
        self.state = TrackState.NEW
        self.frames_lost = 0

    def predict(self):
        self.kf.predict()

    def update(self, bbox_xyxy: np.ndarray, score: float):
        self.kf.update(bbox_xyxy)
        self.score = score
        self.state = TrackState.TRACKED
        self.frames_lost = 0

    @property
    def xyxy(self) -> np.ndarray:
        return self.kf.to_xyxy()


class ByteTracker:
    """
    Manual implementation of the ByteTrack multi-object tracker.

    Args:
        high_thresh: detections with score >= this are "high confidence".
        low_thresh: detections with score in [low_thresh, high_thresh)
                    are "low confidence" and only used in pass 2.
        iou_thresh: minimum IOU to accept a match.
        max_time_lost: how many consecutive frames a track can go
                        unmatched before being permanently removed.
    """

    def __init__(self, high_thresh: float = 0.5, low_thresh: float = 0.1,
                 iou_thresh: float = 0.3, max_time_lost: int = 30):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_thresh = iou_thresh
        self.max_time_lost = max_time_lost

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.frame_id = 0

    def update(self, boxes_xyxy: np.ndarray, scores: np.ndarray) -> List[STrack]:
        """
        Process one frame of detections.

        Args:
            boxes_xyxy: (N, 4) detection boxes.
            scores: (N,) confidence scores.

        Returns:
            list of currently active (TRACKED) STrack objects.
        """
        self.frame_id += 1

        # ---- split detections by confidence ----
        high_mask = scores >= self.high_thresh
        low_mask = (scores >= self.low_thresh) & (~high_mask)

        high_boxes, high_scores = boxes_xyxy[high_mask], scores[high_mask]
        low_boxes, low_scores = boxes_xyxy[low_mask], scores[low_mask]

        # ---- predict new locations for all existing tracks ----
        for t in self.tracked_stracks + self.lost_stracks:
            t.predict()

        # =========================================================
        # PASS 1: high-confidence detections <-> tracked + lost tracks
        # =========================================================
        candidate_tracks = self.tracked_stracks + self.lost_stracks
        track_boxes = np.array([t.xyxy for t in candidate_tracks]) if candidate_tracks else np.empty((0, 4))

        matches, unmatched_tracks, unmatched_high_dets = linear_assignment_iou(
            track_boxes, high_boxes, self.iou_thresh
        )

        matched_ids = set()
        for t_idx, d_idx in matches:
            candidate_tracks[t_idx].update(high_boxes[d_idx], high_scores[d_idx])
            matched_ids.add(id(candidate_tracks[t_idx]))

        # =========================================================
        # PASS 2: remaining unmatched tracks <-> low-confidence detections
        # This is the step that plain SORT does NOT have.
        # =========================================================
        remaining_tracks = [candidate_tracks[i] for i in unmatched_tracks]
        remaining_boxes = np.array([t.xyxy for t in remaining_tracks]) if remaining_tracks else np.empty((0, 4))

        matches2, unmatched_tracks2, _unmatched_low = linear_assignment_iou(
            remaining_boxes, low_boxes, iou_threshold=0.5  # stricter, since low-conf is riskier
        )

        for t_idx, d_idx in matches2:
            remaining_tracks[t_idx].update(low_boxes[d_idx], low_scores[d_idx])
            matched_ids.add(id(remaining_tracks[t_idx]))

        # ---- tracks still unmatched after BOTH passes -> mark lost ----
        still_unmatched_tracks = [remaining_tracks[i] for i in unmatched_tracks2]
        for t in still_unmatched_tracks:
            t.state = TrackState.LOST
            t.frames_lost += 1

        # ---- unmatched high-confidence detections -> spawn new tracks ----
        new_tracks = []
        for d_idx in unmatched_high_dets:
            new_tracks.append(STrack(high_boxes[d_idx], high_scores[d_idx]))
            new_tracks[-1].state = TrackState.TRACKED

        # ---- rebuild the tracked / lost lists ----
        self.tracked_stracks = [
            t for t in candidate_tracks if t.state == TrackState.TRACKED
        ] + new_tracks

        self.lost_stracks = [
            t for t in candidate_tracks
            if t.state == TrackState.LOST and t.frames_lost <= self.max_time_lost
        ]

        # ---- drop tracks lost for too long ----
        self.removed_stracks.extend([
            t for t in candidate_tracks
            if t.state == TrackState.LOST and t.frames_lost > self.max_time_lost
        ])

        return self.tracked_stracks
