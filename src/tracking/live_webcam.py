"""
Real-time Detection + Tracking on a live webcam feed (or any video source),
with a live display window (press 'q' to quit).

Usage:
    python live_webcam.py                      # webcam, live window
    python live_webcam.py --source 0            # same as above, explicit
    python live_webcam.py --source path/to.mp4  # test on a video file instead
    python live_webcam.py --save                # also save the annotated video
"""

import os
# Fixes a common Windows crash: "OMP: Error #15: Initializing libiomp5md.dll,
# but found libiomp5md.dll already initialized." This happens because
# torch (used internally by ultralytics) and numpy/scipy each ship their
# own copy of the OpenMP runtime, and Windows refuses to load two copies
# by default. This must be set BEFORE torch/numpy get imported anywhere,
# which is why it's the very first thing in this file.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from byte_tracker import ByteTracker

COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
]


def color_for_id(track_id: int):
    return COLORS[track_id % len(COLORS)]


def run(source: str, model_name: str = "yolov8n.pt", conf: float = 0.1,
        save: bool = False, output_dir: str = "data/outputs",
        display: bool = True) -> None:
    model = YOLO(model_name)
    tracker = ByteTracker(high_thresh=0.5, low_thresh=0.1,
                           iou_thresh=0.3, max_time_lost=30)

    cap_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video source: {source}. "
            f"If using a webcam, check it's not in use by another app."
        )

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out_path = str(Path(output_dir) / "live_webcam_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_in, (width, height))
        print(f"Saving annotated video to: {out_path}")

    window_name = "Detection + Tracking (press 'q' to quit)"
    gui_available = display
    if display:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except cv2.error:
            print("No GUI backend available (headless environment) -- "
                  "running without a live window.")
            gui_available = False

    frame_count = 0
    t_prev = time.time()
    fps_smooth = 0.0

    print("Running... press 'q' in the video window to stop "
          "(or Ctrl+C in this terminal).")

    try:
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
                label = f"ID {t.track_id}"
                cv2.putText(annotated, label, (x1, max(y1 - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # live FPS counter -- useful to know if this is actually
            # fast enough to call "real-time" on your hardware
            t_now = time.time()
            inst_fps = 1.0 / max(t_now - t_prev, 1e-6)
            fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps if frame_count > 0 else inst_fps
            t_prev = t_now
            cv2.putText(annotated, f"FPS: {fps_smooth:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if writer is not None:
                writer.write(annotated)

            if gui_available:
                cv2.imshow(window_name, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by user ('q' pressed).")
                    break

            frame_count += 1
            if frame_count % 60 == 0:
                print(f"Frame {frame_count} | FPS: {fps_smooth:.1f} | "
                      f"active tracks: {len(active_tracks)}")

    except KeyboardInterrupt:
        print("Stopped by user (Ctrl+C).")

    cap.release()
    if writer is not None:
        writer.release()
    if gui_available:
        cv2.destroyAllWindows()

    print(f"\nDone. Total frames processed: {frame_count}. "
          f"Average FPS: {fps_smooth:.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live webcam detection + tracking")
    parser.add_argument("--source", type=str, default="0",
                         help="'0' for default webcam, or a path to a video file")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--save", action="store_true",
                         help="Also save the annotated video to data/outputs/")
    parser.add_argument("--no-display", dest="display", action="store_false",
                         help="Disable the live window (useful for headless testing)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(source=args.source, model_name=args.model, conf=args.conf,
        save=args.save, display=args.display)
