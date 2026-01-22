"""
Real-time Face + Hand Detection + Gesture Recognition (webcam)

- Uses MediaPipe Holistic for face + hand landmarks
- Uses MediaPipe Tasks GestureRecognizer (loads model bytes to avoid Windows path bug)
- Auto-downloads model if missing (places in ./models/)
- Shows gesture name + confidence, FPS, and debug prints
- Press 'q' to quit  
"""

import os
import time
import math
import requests
import tempfile
import shutil
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "gesture_recognizer.task"
MODEL_PATH = (MODEL_DIR / MODEL_NAME).resolve()

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float32/latest/gesture_recognizer.task"
)

def download_model_if_needed(model_path: Path, url: str):
    """Download model if not present."""
    if model_path.exists():
        print(f"Model already exists at: {model_path}")
        return
    print(f"Downloading model from:\n  {url}\n-> to: {model_path}")
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
    print("Model downloaded successfully.")

# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Utility functions
def normalized_landmarks_to_image_coords(landmark_list, image_width: int, image_height: int):
    """Convert a single hand's normalized landmarks to pixel coordinates."""
    coords = []
    for lm in landmark_list:
        x_px = min(math.floor(lm.x * image_width), image_width - 1)
        y_px = min(math.floor(lm.y * image_height), image_height - 1)
        z_norm = lm.z  
        coords.append((x_px, y_px, z_norm))
    return coords


# Main realtime function
def main():
    # Ensure model exists (download if needed)
    try:
        download_model_if_needed(MODEL_PATH, MODEL_URL)
    except Exception as e:
        print("Model download failed or network error. Please download manually from:")
        print(MODEL_URL)
        print("and place it at:", MODEL_PATH)
        raise

    # Read model bytes and prepare BaseOptions with model_asset_buffer
    with open(MODEL_PATH, "rb") as f:
        model_bytes = f.read()
    print(f"Loaded model bytes from: {MODEL_PATH}")

    # Initialize MediaPipe Holistic (face + hands)
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=True,
    )

    # Prepare Gesture Recognizer using model bytes (avoids Windows path issues)
    base_options = python.BaseOptions(model_asset_buffer=model_bytes)
    gesture_options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,  
        num_hands=2
    )

    try:
        recognizer = vision.GestureRecognizer.create_from_options(gesture_options)
    except Exception as e:
        print("Failed to create GestureRecognizer from options:", e)
        holistic.close()
        raise
    print("Gesture Recognizer loaded successfully.")

    # Initialize webcam — use DirectShow on Windows for better compatibility
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # fallback: try other device index
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Cannot open webcam. Exiting.")
            recognizer.close()
            holistic.close()
            return

    previous_time = 0.0
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Empty frame, ignoring...")
                break

            frame_idx += 1
            # optional: reduce size if you need higher FPS
            frame = cv2.resize(frame, (960, 720))
            image_height, image_width = frame.shape[:2]

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run Holistic to get face + hand landmarks for drawing
            frame_rgb.flags.writeable = False
            holistic_results = holistic.process(frame_rgb)
            frame_rgb.flags.writeable = True

            # Draw face landmarks
            if holistic_results.face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    holistic_results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                )

            # Draw hands from holistic
            if holistic_results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, holistic_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if holistic_results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, holistic_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Prepare MediaPipe Tasks image and timestamp for gesture recognizer
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(frame_rgb))
            timestamp_ms = int(time.time() * 1000)

            # Run gesture recognizer
            try:
                recognition_result = recognizer.recognize_for_video(mp_image, timestamp_ms)
            except Exception as ex:
                print("Gesture recognition error:", ex)
                recognition_result = None

            overlay_texts = []

            if recognition_result:
                # Log gestures
                if recognition_result.gestures:
                    for i, gesture_list in enumerate(recognition_result.gestures):
                        if gesture_list:
                            top_g = gesture_list[0]
                            gesture_name = top_g.category_name
                            confidence = top_g.score
                            print(f"[Frame {frame_idx}] Hand {i}: Gesture='{gesture_name}' conf={confidence:.3f}")
                            overlay_texts.append(f"H{i}:{gesture_name} ({confidence:.2f})")

                # Draw landmarks returned by gesture model (if available)
                if recognition_result.hand_landmarks:
                    for hand_i, hand_landmark_list in enumerate(recognition_result.hand_landmarks):
                        coords = normalized_landmarks_to_image_coords(hand_landmark_list, image_width, image_height)
                        for idx, (x, y, z) in enumerate(coords):
                            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                        xs = [c[0] for c in coords]
                        ys = [c[1] for c in coords]
                        if xs and ys:
                            x_min, x_max = min(xs), max(xs)
                            y_min, y_max = min(ys), max(ys)
                            cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (0, 165, 255), 2)
                            wrist = coords[0]
                            print(f"  Hand {hand_i} wrist (px): x={wrist[0]}, y={wrist[1]}, z={wrist[2]:.4f}")

            # Compose overlay (top-left)
            overlay_y = 30
            for text in overlay_texts:
                cv2.putText(frame, text, (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
                overlay_y += 30

            # FPS
            current_time = time.time()
            fps = 1.0 / (current_time - previous_time) if previous_time != 0 else 0.0
            previous_time = current_time
            cv2.putText(frame, f"{int(fps)} FPS", (image_width - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Show frame
            cv2.imshow("Face+Hands+Gesture (press 'q' to quit)", frame)

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        recognizer.close()
        holistic.close()
        print("Clean exit.")

if __name__ == "__main__":
    main()

