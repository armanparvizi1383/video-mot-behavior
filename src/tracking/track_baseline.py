"""
STEP 2.1 -- Tracking sanity check (using ultralytics' built-in tracker).

Goal: understand what "persistent ID across frames" looks like in practice,
before we implement ByteTrack manually in step 2.2.

This is NOT the final tracker for the project -- it's a quick reference
to validate behavior and compare against our own implementation later.

Usage:
    python track_baseline.py --source ../../data/test_video.mp4 --save
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoids a common Windows OpenMP crash

import argparse
from pathlib import Path
from collections import defaultdict

import cv2
from ultralytics import YOLO


def run_tracking(source: str, model_name: str = "yolov8n.pt",
                  tracker_cfg: str = "bytetrack.yaml",
                  conf: float = 0.35, save: bool = False,
                  output_dir: str = "data/outputs") -> None:
    """
    Run YOLOv8 + built-in ByteTrack and print how IDs persist across frames.
    """
    model = YOLO(model_name)

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
        out_path = str(Path(output_dir) / "tracking_baseline.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"Saving annotated video to: {out_path}")

    # track_id -> number of frames it appeared in (to see how "stable" IDs are)
    id_lifespan = defaultdict(int)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # persist=True keeps track state between calls (same video)
        results = model.track(frame, conf=conf, persist=True,
                               tracker=tracker_cfg, verbose=False)
        result = results[0]

        if result.boxes.id is not None:
            ids = result.boxes.id.int().tolist()
            for tid in ids:
                id_lifespan[tid] += 1

        annotated_frame = result.plot()
        if writer is not None:
            writer.write(annotated_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            active_ids = list(id_lifespan.keys())
            print(f"Frame {frame_count} | active IDs seen so far: {active_ids}")

    cap.release()
    if writer is not None:
        writer.release()

    print("\n--- ID lifespan summary ---")
    print(f"Total unique IDs assigned: {len(id_lifespan)}")
    for tid, n_frames in sorted(id_lifespan.items()):
        print(f"  ID {tid}: appeared in {n_frames} frames")
    print("\nNote: in a good tracker, each real object should keep ONE id")
    print("for its whole time on screen. If you see many short-lived IDs")
    print("for what should be one object, that's an 'ID switch' problem --")
    print("exactly what we'll measure properly with IDF1 in step 2.3.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tracking sanity check (built-in ByteTrack)")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml",
                         help="bytetrack.yaml or botsort.yaml")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tracking(
        source=args.source,
        model_name=args.model,
        tracker_cfg=args.tracker,
        conf=args.conf,
        save=args.save,
    )
