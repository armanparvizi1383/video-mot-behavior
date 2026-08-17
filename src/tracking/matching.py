"""
Matching utilities: compute IOU between boxes, then solve the optimal
assignment problem (tracks <-> detections) with the Hungarian algorithm.

This is the "who goes with who" logic. Given N existing tracks and M new
detections, we build an N x M cost matrix (cost = 1 - IOU) and find the
assignment that minimizes total cost, subject to a max-distance threshold
(so a track will NOT be matched to a detection that barely overlaps it,
even if it's the "best available" option).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Vectorized IOU between two sets of boxes.

    Args:
        boxes_a: (N, 4) array of [x1, y1, x2, y2]
        boxes_b: (M, 4) array of [x1, y1, x2, y2]

    Returns:
        (N, M) IOU matrix.
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=float)

    boxes_a = np.expand_dims(boxes_a, 1)  # (N, 1, 4)
    boxes_b = np.expand_dims(boxes_b, 0)  # (1, M, 4)

    xx1 = np.maximum(boxes_a[..., 0], boxes_b[..., 0])
    yy1 = np.maximum(boxes_a[..., 1], boxes_b[..., 1])
    xx2 = np.minimum(boxes_a[..., 2], boxes_b[..., 2])
    yy2 = np.minimum(boxes_a[..., 3], boxes_b[..., 3])

    w = np.clip(xx2 - xx1, 0, None)
    h = np.clip(yy2 - yy1, 0, None)
    inter = w * h

    area_a = (boxes_a[..., 2] - boxes_a[..., 0]) * (boxes_a[..., 3] - boxes_a[..., 1])
    area_b = (boxes_b[..., 2] - boxes_b[..., 0]) * (boxes_b[..., 3] - boxes_b[..., 1])
    union = area_a + area_b - inter

    return inter / np.clip(union, 1e-6, None)


def linear_assignment_iou(tracks_xyxy: np.ndarray, dets_xyxy: np.ndarray,
                           iou_threshold: float = 0.3):
    """
    Match tracks to detections using IOU + Hungarian algorithm.

    Returns:
        matches: list of (track_idx, det_idx) pairs
        unmatched_tracks: list of track indices with no match
        unmatched_dets: list of detection indices with no match
    """
    if len(tracks_xyxy) == 0 or len(dets_xyxy) == 0:
        return [], list(range(len(tracks_xyxy))), list(range(len(dets_xyxy)))

    iou_matrix = iou_batch(tracks_xyxy, dets_xyxy)
    cost_matrix = 1.0 - iou_matrix  # Hungarian minimizes cost -> maximizes IOU

    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_dets = [], [], []

    for t in range(len(tracks_xyxy)):
        if t not in row_idx:
            unmatched_tracks.append(t)
    for d in range(len(dets_xyxy)):
        if d not in col_idx:
            unmatched_dets.append(d)

    for r, c in zip(row_idx, col_idx):
        if iou_matrix[r, c] < iou_threshold:
            # Hungarian still "assigns" a pair even if it's a bad match --
            # we reject it here if the overlap is too low to be trustworthy.
            unmatched_tracks.append(r)
            unmatched_dets.append(c)
        else:
            matches.append((r, c))

    return matches, unmatched_tracks, unmatched_dets
