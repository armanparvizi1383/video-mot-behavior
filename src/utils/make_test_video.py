"""
Generates a short synthetic video with moving rectangles/circles.
Useful for a quick pipeline sanity check when you don't have a real
video handy yet. NOT a substitute for testing on MOT17 / real footage
later -- YOLO won't detect these shapes as "person"/"car" since they're
not real objects, this is purely to confirm the video I/O works.
"""

import cv2
import numpy as np


def make_test_video(output_path: str = "data/test_video.mp4",
                     width: int = 640, height: int = 480,
                     fps: int = 25, duration_sec: int = 5) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    n_frames = fps * duration_sec
    for i in range(n_frames):
        frame = np.full((height, width, 3), 30, dtype=np.uint8)

        # a moving "person-like" rectangle
        x = int(50 + (width - 150) * (i / n_frames))
        cv2.rectangle(frame, (x, 200), (x + 60, 350), (200, 200, 200), -1)

        # a moving circle ("ball" or generic object)
        cy = int(100 + 50 * np.sin(i / 10))
        cv2.circle(frame, (300, cy), 20, (0, 200, 255), -1)

        writer.write(frame)

    writer.release()
    print(f"Test video written to {output_path} ({n_frames} frames)")


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    make_test_video()
