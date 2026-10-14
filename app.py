"""
Object Detection Model Prototype - Backend Service
Powered by Ultralytics YOLOv8 & FastAPI
"""

import io
import time
import base64
from typing import List, Optional, Dict, Any
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from ultralytics import YOLO

# Initialize FastAPI App
app = FastAPI(
    title="Object Detection Prototype API",
    description="High-performance object detection prototype powered by YOLOv8 and FastAPI",
    version="1.0.0"
)

# Enable CORS for local development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
WEIGHTS_PATH = BASE_DIR / "yolov8n.pt"

# Load YOLO model
print(f"Loading YOLO model from {WEIGHTS_PATH}...")
model = YOLO(str(WEIGHTS_PATH))
class_names = model.names  # Dict mapping class_id (int) -> class_name (str)
print(f"Loaded YOLO model with {len(class_names)} classes.")

# Category Definitions
COCO_PERSON = {"person"}
COCO_ANIMALS = {
    "bird", "cat", "dog", "horse", "sheep", "cow", 
    "elephant", "bear", "zebra", "giraffe"
}
COCO_VEHICLES = {
    "bicycle", "car", "motorcycle", "airplane", "bus", 
    "train", "truck", "boat"
}

def classify_coco_item(class_name: str) -> Dict[str, Any]:
    """Classify a COCO class into Person, Object, or Something Else."""
    name_lower = class_name.lower().strip()
    if name_lower in COCO_PERSON:
        return {
            "category": "person",
            "category_label": "PERSON",
            "display_name": "PERSON",
            "color": "#10b981",       # Emerald Green
            "category_icon": "👤"
        }
    elif name_lower in COCO_ANIMALS:
        return {
            "category": "something_else",
            "category_label": "SOMETHING ELSE",
            "display_name": f"OTHER / ANIMAL: {class_name.title()}",
            "color": "#d946ef",       # Vibrant Fuchsia
            "category_icon": "🐾"
        }
    elif name_lower in COCO_VEHICLES:
        return {
            "category": "something_else",
            "category_label": "SOMETHING ELSE",
            "display_name": f"OTHER / VEHICLE: {class_name.title()}",
            "color": "#06b6d4",       # Cyan
            "category_icon": "🚗"
        }
    else:
        return {
            "category": "object",
            "category_label": "OBJECT",
            "display_name": f"OBJECT: {class_name.title()}",
            "color": "#f59e0b",       # Amber
            "category_icon": "📦"
        }

def get_class_color(class_id: int) -> str:
    cname = class_names.get(class_id, "")
    return classify_coco_item(cname)["color"]


class DetectionFilter(BaseModel):
    confidence: float = 0.25
    iou: float = 0.45
    classes: Optional[List[int]] = None


class Base64DetectRequest(BaseModel):
    image: str  # Base64 data URL or raw base64 string
    confidence: float = 0.25
    iou: float = 0.45
    classes: Optional[List[int]] = None


def run_yolo_inference(
    img_bgr: np.ndarray,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    filter_classes: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Execute YOLO inference on a BGR numpy array and parse detection results."""
    h, w = img_bgr.shape[:2]
    
    t0 = time.perf_counter()
    results = model.predict(
        source=img_bgr,
        conf=conf_thresh,
        iou=iou_thresh,
        classes=filter_classes if filter_classes else None,
        verbose=False
    )
    total_time = (time.perf_counter() - t0) * 1000.0

    result = results[0]
    speed = result.speed  # dict with 'preprocess', 'inference', 'postprocess' in ms

    boxes_data = result.boxes
    detections = []
    class_counts: Dict[str, int] = {}
    category_counts = {"person": 0, "object": 0, "something_else": 0}

    if boxes_data is not None and len(boxes_data) > 0:
        xyxy = boxes_data.xyxy.cpu().numpy()
        confs = boxes_data.conf.cpu().numpy()
        classes = boxes_data.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            box = xyxy[i]
            cid = int(classes[i])
            cname = class_names.get(cid, f"class_{cid}")
            conf = float(confs[i])
            cat_meta = classify_coco_item(cname)

            det = {
                "id": i + 1,
                "class_id": cid,
                "class_name": cname,
                "category": cat_meta["category"],
                "category_label": cat_meta["category_label"],
                "display_name": cat_meta["display_name"],
                "category_icon": cat_meta["category_icon"],
                "confidence": round(conf, 4),
                "box": {
                    "x1": round(float(box[0]), 1),
                    "y1": round(float(box[1]), 1),
                    "x2": round(float(box[2]), 1),
                    "y2": round(float(box[3]), 1),
                    "width": round(float(box[2] - box[0]), 1),
                    "height": round(float(box[3] - box[1]), 1),
                },
                "color": cat_meta["color"]
            }
            detections.append(det)
            class_counts[cname] = class_counts.get(cname, 0) + 1
            category_counts[cat_meta["category"]] += 1

    # Sort detections by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    return {
        "detections": detections,
        "total_detections": len(detections),
        "class_counts": class_counts,
        "category_counts": category_counts,
        "image_size": {"width": w, "height": h},
        "metrics": {
            "preprocess_ms": round(speed.get("preprocess", 0.0), 2),
            "inference_ms": round(speed.get("inference", 0.0), 2),
            "postprocess_ms": round(speed.get("postprocess", 0.0), 2),
            "total_ms": round(total_time, 2),
            "fps": round(1000.0 / total_time if total_time > 0 else 0, 1)
        }
    }


@app.get("/api/info")
async def get_model_info():
    """Return model architecture info, device status, and supported COCO classes."""
    return {
        "model_name": "YOLOv8 Nano (yolov8n)",
        "task": "Object Detection",
        "dataset": "MS COCO (80 Classes)",
        "total_classes": len(class_names),
        "classes": [{"id": k, "name": v, "color": get_class_color(k)} for k, v in class_names.items()],
        "defaults": {
            "confidence": 0.25,
            "iou": 0.45
        }
    }


@app.post("/api/detect")
async def detect_uploaded_image(
    file: UploadFile = File(...),
    confidence: float = Form(0.25),
    iou: float = Form(0.45),
    classes: Optional[str] = Form(None)
):
    """Detect objects in an uploaded image file."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return JSONResponse(status_code=400, content={"error": "Failed to decode uploaded image."})

    filter_classes = None
    if classes:
        try:
            filter_classes = [int(c.strip()) for c in classes.split(",") if c.strip()]
        except ValueError:
            pass

    results = run_yolo_inference(
        img_bgr=img_bgr,
        conf_thresh=confidence,
        iou_thresh=iou,
        filter_classes=filter_classes
    )
    return results


@app.post("/api/detect-base64")
async def detect_base64_frame(payload: Base64DetectRequest):
    """Detect objects in a base64 encoded image frame (ideal for webcam capture)."""
    raw_b64 = payload.image
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(raw_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return JSONResponse(status_code=400, content={"error": "Unable to decode base64 image data."})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Base64 decoding error: {str(e)}"})

    results = run_yolo_inference(
        img_bgr=img_bgr,
        conf_thresh=payload.confidence,
        iou_thresh=payload.iou,
        filter_classes=payload.classes
    )
    return results


@app.websocket("/ws/live")
async def websocket_live_detection(websocket: WebSocket):
    """Real-time live detection over WebSocket for responsive webcam feeds."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            raw_b64 = data.get("image", "")
            conf = float(data.get("confidence", 0.25))
            iou = float(data.get("iou", 0.45))
            filter_classes = data.get("classes", None)

            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]

            if not raw_b64:
                await websocket.send_json({"error": "Empty frame"})
                continue

            img_bytes = base64.b64decode(raw_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img_bgr is not None:
                results = run_yolo_inference(img_bgr, conf, iou, filter_classes)
                await websocket.send_json(results)
            else:
                await websocket.send_json({"error": "Failed to decode frame"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")


# Serve static web frontend
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(STATIC_DIR / "index.html")

if __name__ == "__main__":
    import uvicorn
    import sys
    import os

    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    else:
        port = int(os.environ.get("PORT", 8000))

    print(f"\n[Starting Server] Open http://127.0.0.1:{port} in your browser\n")
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=True)
