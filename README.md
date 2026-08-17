# Video Multi-Object Tracking & Behavior Analysis

Detect, track, and analyze the behavior of multiple objects in video.

**Status:** 🚧 Step 1 — Detection baseline working.


<img width="640" height="360" alt="demo" src="https://github.com/user-attachments/assets/a1605443-4665-4b71-a48a-ffc13f0f617a" />




## Roadmap
- [x] Step 1: Detection baseline (YOLOv8 on a video/webcam)
- [x] Step 2: Multi-object tracking
  - [x] 2.1: Sanity check with ultralytics' built-in ByteTrack
  - [x] 2.2: Manual ByteTrack implementation (Kalman filter + two-stage
        IOU matching, from scratch) — unit tested
  - [ ] 2.3: Quantitative evaluation on MOT17 (MOTA/IDF1) — script ready,
        run it yourself locally (see below, dataset host is blocked in
        the sandbox this was built in)
- [ ] Step 3: Re-ID to reduce ID switches under occlusion
- [ ] Step 4: Behavior analysis (LSTM over trajectories)
- [ ] Step 5: Real-time demo (Gradio) + deployment

### Result so far
On our (synthetic) sanity-check video, the built-in tracker produced
**5 unique IDs** for 2 real moving objects (ID switches). Our own
ByteTrack implementation reduced this to **3 IDs** on the same video —
demonstrating that the two-stage high/low-confidence matching pass is
doing its job. Full MOTA/IDF1 numbers on MOT17 pending (see below).

## Project structure
```
video-mot-behavior/
├── src/
│   ├── detection/      # YOLOv8 detection
│   ├── tracking/        # ByteTrack / Re-ID (coming next)
│   ├── behavior/        # behavior classification (coming later)
│   └── utils/            # helpers (test video generation, etc.)
├── data/                 # videos + outputs (gitignored, except samples)
├── notebooks/            # exploration / evaluation notebooks
├── configs/              # config files (model params, paths)
└── tests/                # unit tests
```

## Quickstart

```bash
pip install -r requirements.txt

# generate a quick synthetic test video (no real dataset needed)
python src/utils/make_test_video.py

# run baseline detection
python src/detection/detect_baseline.py --source data/test_video.mp4 --save
```

Output video is written to `data/outputs/detection_baseline.mp4`.

To use a real video or webcam:
```bash
python src/detection/detect_baseline.py --source path/to/your_video.mp4 --save
python src/detection/detect_baseline.py --source 0   # webcam
```

### Step 2 usage

```bash
# 2.1 - sanity check with ultralytics' built-in tracker
python src/tracking/track_baseline.py --source data/test_video.mp4 --save

# 2.2 - our own manual ByteTrack (Kalman filter + 2-stage IOU matching)
python src/tracking/track_custom.py --source data/test_video.mp4 --save

# run unit tests for the matching logic
python -m pytest tests/test_matching.py -v
```

### Step 2.3 — evaluating on MOT17 (run locally)

The MOT17 dataset host (`motchallenge.net`) isn't reachable from the
sandboxed environment this project was built in, so this step needs to
be run on your own machine:

1. Download a sequence from https://motchallenge.net/data/MOT17/
   (e.g. `MOT17-04-DPM`, or the full `MOT17.zip`) and unzip it.
2. Run:
   ```bash
   pip install -r requirements.txt
   cd src/tracking
   python eval_mot17.py --seq_dir /path/to/MOT17/train/MOT17-04-DPM
   ```
3. This prints MOTA, MOTP, IDF1, ID switches, false positives, and
   misses — the standard metrics used to benchmark trackers.

Once you have real numbers, add a comparison table here, e.g.:

| Tracker                  | MOTA | IDF1 | ID switches |
|---------------------------|------|------|--------------|
| Built-in ByteTrack (2.1)  |      |      |              |
| Our ByteTrack (2.2)       |      |      |              |

## Next step
Step 3: add Re-ID embeddings to further reduce ID switches under heavy occlusion.
