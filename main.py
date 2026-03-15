from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import uuid

from functions import load_model, detect_on_video, is_valid_video_file

app = FastAPI(
    title="Violence Detection API",
    version="1.0.0",
    description="Upload a video and detect violence using a trained YOLO model."
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    model = load_model(str(MODEL_PATH))
    model_error = None
except Exception as e:
    model = None
    model_error = str(e)


@app.get("/")
def health_check():
    return {
        "status": "OK",
        "message": "Violence Detection API is running"
    }


@app.get("/info")
def info():
    return {
        "project": "Violence Detection using YOLO",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "model_error": model_error
    }


@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model failed to load: {model_error}")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    if not is_valid_video_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Invalid video type. Allowed: .mp4, .avi, .mov, .mkv"
        )

    unique_id = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{unique_id}_{file.filename}"
    output_filename = f"detected_{unique_id}.mp4"
    output_path = OUTPUT_DIR / output_filename

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = detect_on_video(
            model=model,
            input_path=str(input_path),
            output_path=str(output_path)
        )

        return {
            "message": "Detection finished successfully",
            "output_file": output_filename,
            "download_url": f"/download/{output_filename}",
            "video_info": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")

    finally:
        if input_path.exists():
            input_path.unlink(missing_ok=True)


@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found.")

    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=filename
    )