"""
STEP 2.3 -- Quantitative evaluation against MOT17 ground truth.

*** RUN THIS ON YOUR OWN MACHINE, NOT IN THIS SANDBOX ***
(the sandbox this was built in cannot reach motchallenge.net)

--------------------------------------------------------------------
How to get the data (do this yourself, one time):
--------------------------------------------------------------------
1. Go to https://motchallenge.net/data/MOT17/ and download "MOT17.zip"
   (or just the smaller MOT17-04 / MOT17-09 sequence if you want a
   quick single-sequence test instead of the full ~5GB set).
2. Unzip it. You should get a structure like:
       MOT17/train/MOT17-04-DPM/img1/*.jpg
       MOT17/train/MOT17-04-DPM/gt/gt.txt
3. Point --seq_dir at one sequence folder, e.g.:
       python eval_mot17.py --seq_dir path/to/MOT17/train/MOT17-04-DPM

--------------------------------------------------------------------
What this script does:
--------------------------------------------------------------------
1. Runs YOLO + our ByteTracker over every frame in img1/
2. Formats our predictions the way motmetrics expects
3. Loads gt/gt.txt (ground truth track annotations)
4. Computes MOTA, MOTP, IDF1, ID switches, etc.
--------------------------------------------------------------------
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoids a common Windows OpenMP crash

import argparse
from pathlib import Path

import cv2
import numpy as np
import motmetrics as mm
from ultralytics import YOLO

from byte_tracker import ByteTracker
from matching import iou_batch


def run_tracker_on_sequence(seq_dir: Path, model_name: str = "yolov8n.pt"):
    """Run detection + tracking on all frames in seq_dir/img1, return predictions."""
    model = YOLO(model_name)
    tracker = ByteTracker(high_thresh=0.5, low_thresh=0.1,
                           iou_thresh=0.3, max_time_lost=30)

    img_dir = seq_dir / "img1"
    frame_paths = sorted(img_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {img_dir}")

    # predictions[frame_idx] = list of (track_id, x1, y1, x2, y2)
    predictions = {}

    for i, frame_path in enumerate(frame_paths, start=1):
        frame = cv2.imread(str(frame_path))

        # MOT17 evaluation traditionally focuses on the "person" class
        results = model.predict(frame, conf=0.1, classes=[0], verbose=False)
        result = results[0]

        if len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
        else:
            boxes_xyxy = np.empty((0, 4))
            scores = np.empty((0,))

        active_tracks = tracker.update(boxes_xyxy, scores)
        predictions[i] = [(t.track_id, *t.xyxy) for t in active_tracks]

        if i % 50 == 0:
            print(f"  processed {i}/{len(frame_paths)} frames")

    return predictions


def load_mot17_gt(gt_path: Path):
    """
    MOT17 gt.txt format (comma-separated, no header):
        frame, id, x, y, w, h, conf, class, visibility

    We only keep conf==1 (valid annotation) and class==1 (pedestrian),
    which is the standard MOT17 evaluation convention.
    """
    gt = {}
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            frame, obj_id = int(parts[0]), int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            conf, cls = int(float(parts[6])), int(float(parts[7]))

            if conf != 1 or cls != 1:
                continue

            gt.setdefault(frame, []).append((obj_id, x, y, x + w, y + h))
    return gt


def evaluate(predictions: dict, ground_truth: dict) -> mm.MOTAccumulator:
    """Feed frame-by-frame predictions and GT into motmetrics."""
    acc = mm.MOTAccumulator(auto_id=True)

    all_frames = sorted(set(predictions.keys()) | set(ground_truth.keys()))
    for frame in all_frames:
        gt_objs = ground_truth.get(frame, [])
        pred_objs = predictions.get(frame, [])

        gt_ids = [o[0] for o in gt_objs]
        gt_boxes = np.array([o[1:] for o in gt_objs]) if gt_objs else np.empty((0, 4))

        pred_ids = [o[0] for o in pred_objs]
        pred_boxes = np.array([o[1:] for o in pred_objs]) if pred_objs else np.empty((0, 4))

        # motmetrics wants an IOU-based distance matrix (NaN = no match possible).
        # NOTE: we build this ourselves with our own iou_batch() from matching.py
        # instead of calling motmetrics.distances.iou_matrix(), because that
        # function uses np.asfarray() internally, which was removed in NumPy 2.0
        # and breaks under recent numpy/opencv combos. Same math, no dependency
        # headache.
        max_iou = 0.5
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            dist_matrix = np.empty((0, 0))
        else:
            iou = iou_batch(gt_boxes, pred_boxes)
            dist_matrix = 1.0 - iou
            dist_matrix[iou < (1.0 - max_iou)] = np.nan

        acc.update(gt_ids, pred_ids, dist_matrix)

    return acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate ByteTracker on a MOT17 sequence")
    parser.add_argument("--seq_dir", type=str, required=True,
                         help="Path to a MOT17 sequence folder, e.g. MOT17/train/MOT17-04-DPM")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)
    gt_path = seq_dir / "gt" / "gt.txt"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found at {gt_path}")

    print(f"Running tracker on {seq_dir} ...")
    predictions = run_tracker_on_sequence(seq_dir, args.model)

    print("Loading ground truth ...")
    ground_truth = load_mot17_gt(gt_path)

    print("Computing metrics ...")
    acc = evaluate(predictions, ground_truth)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["mota", "motp", "idf1", "num_switches", "num_false_positives",
                 "num_misses", "num_fragmentations"],
        name=seq_dir.name,
    )
    print("\n" + mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    ))


if __name__ == "__main__":
    main()
