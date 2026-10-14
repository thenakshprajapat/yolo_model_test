# OmniVision AI — Object Detection Prototype

An end-to-end, high-performance object detection prototype built with **YOLOv8**, **FastAPI**, and a **modern dark glassmorphism interactive web dashboard**.

> 💡 **New to AI or Computer Vision?** Check out the [Beginner's Guide (Explained in Plain English)](file:///d:/makerspace/odm/BEGINNER_GUIDE.md) to understand how the entire pipeline works step-by-step!

---

## 🚀 Features

- **Real-Time Camera Detection (Person vs Object vs Something Else)**:
  - **Person** (👤 Emerald Green): Detects people/humans in real-time.
  - **Object** (📦 Amber Gold): Inanimate items like cell phones, laptops, cups, bottles, backpacks, chairs, books, etc.
  - **Something Else** (⚡ / 🐾 Vibrant Magenta / Cyan): Animals (dogs, cats, birds, horses, etc.) and vehicles (cars, bicycles, buses, etc.).
- **Two Real-Time Camera Modes**:
  - **1. Standalone Native Desktop Camera (`live_cam.py`)**: Zero-latency direct OpenCV camera window with HUD telemetry, real-time FPS counter, high-contrast corner brackets, and snapshot capture (`S` key).
  - **2. Interactive Web Dashboard (`app.py`)**: Real-time webcam streaming via WebSockets/HTTP, live Classification Matrix cards, bi-directional bounding box inspection, and adjustable thresholds.
- **Pretrained YOLOv8 Nano Architecture**:
  - Pre-trained on 80 standard MS COCO classes.
  - Lightweight (~6.2 MB) with low CPU inference latency.
- **REST & WebSocket API**:
  - `GET /api/info`: Model metadata, classes list, default thresholds.
  - `POST /api/detect`: Multipart image upload detection endpoint.
  - `POST /api/detect-base64`: Base64 frame detection endpoint.
  - `WebSocket /ws/live`: Low-latency bi-directional streaming for live webcam detection.

---

## 📂 Project Structure

```
odm/
├── live_cam.py                # Standalone real-time camera detector (OpenCV + YOLO)
├── app.py                     # FastAPI application server (REST + WebSocket)
├── detect_cli.py              # Command-line detection interface for files/videos
├── test_api.py                # Automated verification script
├── yolov8n.pt                 # YOLOv8 Nano pre-trained weights
├── static/
│   ├── index.html             # Modern UI layout with Classification Matrix
│   ├── css/
│   │   └── style.css          # Glassmorphism dark-theme styling
│   ├── js/
│   │   └── app.js             # Canvas rendering, camera stream & state logic
│   └── samples/
│       ├── street_traffic.jpg # Sample preset 1: Busy city street
│       └── living_room.jpg    # Sample preset 2: Cozy living room with pets
└── README.md                  # Project documentation
```

---

## 🏃 Quick Start

### 1. Run Real-Time Camera Detection (Native Window)

For instant zero-latency camera detection directly on your screen:
```bash
python live_cam.py
```
- **Controls**:
  - `Q` or `ESC`: Quit
  - `S`: Save a snapshot with bounding boxes to disk
  - `+` / `-`: Increase or decrease confidence threshold

*(Optional: select a specific camera index if you have multiple cameras with `--camera 1`)*

### 2. Launch the Web Interface

Start the local server:
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Then open your web browser at:
**[http://127.0.0.1:8000](http://127.0.0.1:8000)**

Click the **"Live Webcam"** tab and tap **"Launch Webcam Detection"** to see live video with the real-time **Classification Matrix**!

### 3. Run CLI Detection on Images / Videos

```bash
python detect_cli.py --source static/samples/living_room.jpg --conf 0.3
```

---

## 🔮 Next Steps: Custom Model Fine-Tuning

When you are ready to transition from this general prototype to custom object classes:
1. **Prepare Dataset**: Structure images and annotation labels in YOLO format (`images/train`, `labels/train`, `data.yaml`).
2. **Fine-tune with Python**:
   ```python
   from ultralytics import YOLO

   model = YOLO("yolov8n.pt")
   model.train(data="custom_dataset.yaml", epochs=50, imgsz=640)
   ```
3. **Swap Weights**: Point `app.py` to `runs/detect/train/weights/best.pt`.
