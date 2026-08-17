"""
STEP 2.2 -- Run detection + OUR OWN manual ByteTrack implementation.

This is the "real" version -- not the library's built-in tracker from
step 2.1, but our own Kalman filter + two-stage IOU matching.

Usage:
    python track_custom.py --source ../../data/test_video.mp4 --save
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoids a common Windows OpenMP crash

import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

from byte_tracker import ByteTracker

# distinct colors so each ID is visually easy to follow across frames
COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
]


def color_for_id(track_id: int):
    return COLORS[track_id % len(COLORS)]


def run(source: str, model_name: str = "yolov8n.pt", conf: float = 0.1,
        save: bool = False, output_dir: str = "data/outputs") -> None:
    """
    Args:
        conf: we deliberately use a LOW confidence here (e.g. 0.1) because
        our own ByteTracker needs BOTH high and low confidence detections
        to do its two-stage matching -- filtering them out before they
        reach the tracker would defeat the purpose.
    """
    model = YOLO(model_name)
    tracker = ByteTracker(high_thresh=0.5, low_thresh=0.1,
                           iou_thresh=0.3, max_time_lost=30)

    cap_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out_path = str(Path(output_dir) / "tracking_custom_bytetrack.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"Saving annotated video to: {out_path}")

    id_lifespan = defaultdict(int)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf, verbose=False)
        result = results[0]

        if len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
        else:
            boxes_xyxy = np.empty((0, 4))
            scores = np.empty((0,))

        active_tracks = tracker.update(boxes_xyxy, scores)

        annotated = frame.copy()
        for t in active_tracks:
            x1, y1, x2, y2 = t.xyxy.astype(int)
            color = color_for_id(t.track_id)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"ID {t.track_id} ({t.score:.2f})"
            cv2.putText(annotated, label, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            id_lifespan[t.track_id] += 1

        if writer is not None:
            writer.write(annotated)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Frame {frame_count} | active IDs: "
                  f"{[t.track_id for t in active_tracks]}")

    cap.release()
    if writer is not None:
        writer.release()

    print("\n--- ID lifespan summary (our custom ByteTrack) ---")
    print(f"Total unique IDs assigned: {len(id_lifespan)}")
    for tid, n_frames in sorted(id_lifespan.items()):
        print(f"  ID {tid}: appeared in {n_frames} frames")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Custom ByteTrack demo")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(source=args.source, model_name=args.model, conf=args.conf, save=args.save)
