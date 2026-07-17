"""Real-time hand-pose tracking from the webcam using MediaPipe's hand_landmarker.task.

Run it on a machine with a camera and display:

    python app.py                          # auto-find hand_landmarker.task
    python app.py --model path/to/hand_landmarker.task
    python app.py --camera 1 --num-hands 2

Controls: press 'q' or ESC to quit.

Requires: mediapipe, opencv-python, numpy.
    pip install mediapipe opencv-python numpy
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# MediaPipe Hands 21-joint connectivity.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm edge
)

# Where to look for the model if --model isn't given.
MODEL_CANDIDATES = [
    Path(__file__).resolve().parent / "hand_landmarker.task",
    Path.home() / "datasets" / "hand_landmarker.task",
    Path.home() / "hand-pose" / "models" / "hand_landmarker.task",
]


def locate_model(explicit: str | None) -> str:
    if explicit:
        if not Path(explicit).is_file():
            raise SystemExit(f"model not found: {explicit}")
        return explicit
    for cand in MODEL_CANDIDATES:
        if cand.is_file():
            return str(cand)
    raise SystemExit(
        "Could not find hand_landmarker.task. Pass --model /path/to/hand_landmarker.task "
        "or download it:\n  curl -fsSL -o hand_landmarker.task "
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    )


def make_landmarker(model_path: str, num_hands: int):
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def draw_hands(frame: np.ndarray, result) -> None:
    h, w = frame.shape[:2]
    hands = getattr(result, "hand_landmarks", None) or []
    handedness = getattr(result, "handedness", None) or []

    for idx, lm in enumerate(hands):
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 220, 0), 2)
        for i, (x, y) in enumerate(pts):
            color = (0, 220, 255) if i == 0 else (32, 32, 255)  # wrist vs joints (BGR)
            cv2.circle(frame, (x, y), 4, color, -1)

        if idx < len(handedness) and handedness[idx]:
            cat = handedness[idx][0]
            # The frame is mirrored (cv2.flip) before detection, so MediaPipe's
            # Left/Right is reversed relative to the user's real hand. Swap it back.
            true_name = {"Left": "Right", "Right": "Left"}.get(
                cat.category_name, cat.category_name)
            label = f"{true_name} {cat.score:.2f}"
            cv2.putText(frame, label, (pts[0][0] - 10, pts[0][1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="path to hand_landmarker.task")
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    model_path = locate_model(args.model)
    print(f"using model: {model_path}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")

    landmarker = make_landmarker(model_path, args.num_hands)
    start = time.time()
    prev = start
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("failed to read frame")
                break

            frame = cv2.flip(frame, 1)  # mirror for a natural selfie view
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            draw_hands(frame, result)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now
            cv2.putText(frame, f"{fps:4.1f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Hand Pose Tracking (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
