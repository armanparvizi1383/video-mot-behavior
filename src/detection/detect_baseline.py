"""
Baseline object detection script using YOLOv8 (pretrained on COCO).

Purpose: sanity-check that the pipeline works end-to-end before adding
tracking and behavior analysis on top.

Usage:
    python detect_baseline.py --source path/to/video.mp4
    python detect_baseline.py --source 0            # webcam
    python detect_baseline.py --source path/to/video.mp4 --save
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoids a common Windows OpenMP crash

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def run_detection(source: str, model_name: str = "yolov8n.pt",
                   conf: float = 0.35, save: bool = False,
                   output_dir: str = "data/outputs") -> None:
    """
    Run YOLOv8 detection on a video source and display/save annotated frames.

    Args:
        source: path to video file, or "0" for webcam.
        model_name: which YOLOv8 checkpoint to use (n=nano, s=small, ...).
        conf: confidence threshold for detections.
        save: whether to save the annotated video to disk.
        output_dir: where to save the output video if save=True.
    """
    model = YOLO(model_name)  # auto-downloads weights on first run

    # webcam is passed as a string "0" from argparse; convert to int
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
        out_path = str(Path(output_dir) / "detection_baseline.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"Saving annotated video to: {out_path}")

    frame_count = 0
    total_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference on this single frame
        results = model.predict(frame, conf=conf, verbose=False)
        result = results[0]

        total_detections += len(result.boxes)
        annotated_frame = result.plot()  # draws boxes + labels + confidence

        if writer is not None:
            writer.write(annotated_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames | "
                  f"detections this frame: {len(result.boxes)}")

    cap.release()
    if writer is not None:
        writer.release()

    print("\n--- Summary ---")
    print(f"Total frames processed : {frame_count}")
    print(f"Total detections       : {total_detections}")
    print(f"Avg detections / frame : {total_detections / max(frame_count, 1):.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 baseline detection")
    parser.add_argument("--source", type=str, required=True,
                         help="Path to video file, or '0' for webcam")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                         help="YOLOv8 checkpoint (yolov8n/s/m/l/x.pt)")
    parser.add_argument("--conf", type=float, default=0.35,
                         help="Confidence threshold")
    parser.add_argument("--save", action="store_true",
                         help="Save annotated video to data/outputs/")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_detection(
        source=args.source,
        model_name=args.model,
        conf=args.conf,
        save=args.save,
    )
