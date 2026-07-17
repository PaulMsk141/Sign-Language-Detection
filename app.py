"""Real-time hand-pose tracking from the webcam using MediaPipe's hand_landmarker.task.

Run it on a machine with a camera and display:

    python app.py                          # auto-find hand_landmarker.task
    python app.py --model path/to/hand_landmarker.task
    python app.py --camera 1 --num-hands 2

It also loads a trained classifier and shows the predicted ASL letter (A-Z) for
each detected hand. By default it uses the neural network (neural_network/asl_mlp.joblib)
and falls back to the logistic-regression model if the neural net is missing.
Use --classifier to point at any other .joblib model.

Controls: press 'q' or ESC to quit.

Requires: mediapipe, opencv-python, numpy, scikit-learn, joblib.
    pip install mediapipe opencv-python numpy scikit-learn joblib
"""

from __future__ import annotations

import argparse
import time
import warnings
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

# Trained classifiers, in priority order. The neural network is the default;
# the logistic-regression model is the fallback.
_HERE = Path(__file__).resolve().parent
CLASSIFIER_CANDIDATES = [
    _HERE / "neural_network" / "asl_mlp.joblib",
    _HERE / "asl_mlp.joblib",
    _HERE / "logistic_regression" / "asl_logreg.joblib",
    _HERE / "asl_logreg.joblib",
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


def load_classifier(explicit: str | None):
    """Load the ASL classifier (neural net by default), or None if unavailable."""
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            print(f"note: classifier not found at {path} - letters will not be shown")
            return None
    else:
        path = next((c for c in CLASSIFIER_CANDIDATES if c.is_file()), None)
        if path is None:
            print("note: no classifier (.joblib) found - letters will not be shown")
            return None
    try:
        import joblib
    except ImportError:
        print("note: joblib/scikit-learn not installed - letters will not be shown "
              "(pip install scikit-learn joblib)")
        return None
    clf = joblib.load(path)
    kind = type(clf).__name__
    print(f"using classifier: {path.name} ({kind})  classes: {''.join(clf.classes_)}")
    return clf


def predict_letter(clf, lm) -> tuple[str, float] | None:
    """Map 21 landmarks (x, y, z each) to a letter using the classifier."""
    if clf is None:
        return None
    # Feature order matches the notebook: x0, y0, z0, x1, y1, z1, ... x20, y20, z20.
    feats = np.array([[c for p in lm for c in (p.x, p.y, p.z)]], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # model was fit on named columns; array is fine
        letter = str(clf.predict(feats)[0])
        score = float(clf.predict_proba(feats).max()) if hasattr(clf, "predict_proba") else 0.0
    return letter, score


def draw_hands(frame: np.ndarray, result, clf=None) -> None:
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

        pred = predict_letter(clf, lm)
        if pred is not None:
            letter, score = pred
            text = f"{letter} {score:.2f}"
            x = max(pts[9][0] - 20, 10)   # near the middle-finger base
            y = max(min(pts[9][1], frame.shape[0] - 10), 40)
            cv2.putText(frame, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
            cv2.putText(frame, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="path to hand_landmarker.task")
    parser.add_argument("--classifier", default=None,
                        help="path to a .joblib model (default: neural_network/asl_mlp.joblib, "
                             "falls back to the logistic-regression model)")
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    model_path = locate_model(args.model)
    print(f"using model: {model_path}")
    classifier = load_classifier(args.classifier)

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

            draw_hands(frame, result, classifier)

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
