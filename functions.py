import cv2
from ultralytics import YOLO
from pathlib import Path


def load_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return YOLO(model_path)


def is_valid_video_file(filename: str) -> bool:
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    return Path(filename).suffix.lower() in allowed_extensions


def detect_on_video(model, input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError("Error opening video file.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        out.write(annotated_frame)
        processed_frames += 1

    cap.release()
    out.release()

    return {
        "input_path": input_path,
        "output_path": output_path,
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "fps": fps,
        "width": width,
        "height": height
    }