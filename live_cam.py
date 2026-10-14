"""
Real-Time Camera Detection: Person vs Object vs Something Else
Powered by YOLOv8 and OpenCV

Usage:
    python live_cam.py
    python live_cam.py --camera 0 --conf 0.35
    python live_cam.py --weights yolov8n.pt
"""

import argparse
import time
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

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

# Color Scheme (BGR for OpenCV)
COLOR_PERSON = (46, 204, 113)     # Emerald Green
COLOR_OBJECT = (11, 158, 245)     # Amber / Gold
COLOR_OTHER_ANIMAL = (239, 70, 217) # Vibrant Magenta
COLOR_OTHER_VEHICLE = (212, 182, 6) # Cyan / Blue
COLOR_BG_DARK = (24, 24, 27)      # Dark Slate for HUD

def classify_detection(class_name: str):
    name_lower = class_name.lower().strip()
    if name_lower in COCO_PERSON:
        return {
            "category": "PERSON",
            "display_name": "PERSON",
            "color": COLOR_PERSON,
            "badge_key": "person"
        }
    elif name_lower in COCO_ANIMALS:
        return {
            "category": "SOMETHING ELSE",
            "display_name": f"OTHER / ANIMAL: {class_name.title()}",
            "color": COLOR_OTHER_ANIMAL,
            "badge_key": "other"
        }
    elif name_lower in COCO_VEHICLES:
        return {
            "category": "SOMETHING ELSE",
            "display_name": f"OTHER / VEHICLE: {class_name.title()}",
            "color": COLOR_OTHER_VEHICLE,
            "badge_key": "other"
        }
    else:
        return {
            "category": "OBJECT",
            "display_name": f"OBJECT: {class_name.title()}",
            "color": COLOR_OBJECT,
            "badge_key": "object"
        }

def draw_hud(frame, fps: float, counts: dict, conf_thresh: float):
    """Draw top header HUD overlay with real-time telemetry and category counts."""
    h, w = frame.shape[:2]
    hud_h = 52
    
    # Semi-transparent HUD background bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), COLOR_BG_DARK, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Bottom border line for HUD
    cv2.line(frame, (0, hud_h), (w, hud_h), (60, 60, 70), 1)

    # Title & FPS
    title_text = "OmniVision Real-Time Detector"
    fps_text = f"FPS: {fps:.1f} | Conf: {int(conf_thresh * 100)}%"
    cv2.putText(frame, title_text, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, fps_text, (15, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 175), 1, cv2.LINE_AA)

    # Category Badges on the right
    badges = [
        (f"PERSONS: {counts['person']}", COLOR_PERSON),
        (f"OBJECTS: {counts['object']}", COLOR_OBJECT),
        (f"SOMETHING ELSE: {counts['other']}", COLOR_OTHER_ANIMAL),
    ]

    offset_x = w - 20
    for text, color in reversed(badges):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        bx1 = offset_x - tw - 18
        bx2 = offset_x
        by1 = 12
        by2 = 40
        
        # Draw pill
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (40, 40, 45), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 1, cv2.LINE_AA)
        
        # Color dot
        cv2.circle(frame, (bx1 + 10, by1 + 14), 4, color, -1, cv2.LINE_AA)
        # Text
        cv2.putText(frame, text, (bx1 + 20, by1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        offset_x = bx1 - 10

def draw_corner_rect(img, x1, y1, x2, y2, color, thickness=2, corner_len=14):
    """Draw stylish bounding box with enhanced corner markers."""
    # Main rectangle with thin line
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    
    # Thicker corners for a modern HUD look
    # Top-Left
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness, cv2.LINE_AA)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness, cv2.LINE_AA)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness, cv2.LINE_AA)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness, cv2.LINE_AA)

def main():
    parser = argparse.ArgumentParser(description="Real-Time Camera Person vs Object vs Something Else Detector")
    parser.add_argument("--camera", type=int, default=0, help="Camera / webcam device index (default: 0)")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Path to YOLOv8 weights")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (0.1 - 0.9)")
    parser.add_argument("--width", type=int, default=1280, help="Target camera resolution width")
    parser.add_argument("--height", type=int, default=720, help="Target camera resolution height")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[Error] Weights file '{args.weights}' not found.")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f"  Real-Time Camera Detection: Person | Object | Other")
    print(f"  Camera Index : {args.camera}")
    print(f"  Model Weights: {args.weights}")
    print(f"  Confidence   : {args.conf}")
    print(f"=======================================================")
    print("Controls:")
    print("  [Q] or [ESC] : Quit")
    print("  [S]          : Save snapshot image to disk")
    print("  [+] / [-]    : Adjust confidence threshold")
    print("=======================================================\n")

    print("[Info] Loading YOLO model...")
    model = YOLO(str(weights_path))
    class_names = model.names
    print(f"[Info] Loaded model with {len(class_names)} classes.")

    print(f"[Info] Opening camera device {args.camera}...")
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    
    if not cap.isOpened():
        # Fallback without CAP_DSHOW
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"[Error] Could not open camera {args.camera}. Ensure camera is connected and not in use by another app.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    conf_thresh = args.conf
    prev_time = time.perf_counter()
    fps_smooth = 0.0

    window_name = "Real-Time Camera Detector (Person | Object | Something Else)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[Warning] Failed to grab frame from camera.")
                time.sleep(0.05)
                continue

            curr_time = time.perf_counter()
            delta = curr_time - prev_time
            prev_time = curr_time
            if delta > 0:
                current_fps = 1.0 / delta
                fps_smooth = current_fps if fps_smooth == 0.0 else (fps_smooth * 0.85 + current_fps * 0.15)

            # YOLO inference
            results = model.predict(source=frame, conf=conf_thresh, verbose=False)
            result = results[0]

            counts = {"person": 0, "object": 0, "other": 0}

            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)

                for i in range(len(xyxy)):
                    cid = int(classes[i])
                    cname = class_names.get(cid, "item")
                    conf = float(confs[i])
                    info = classify_detection(cname)
                    counts[info["badge_key"]] += 1

                    box = xyxy[i]
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    color = info["color"]

                    # Draw stylish corner box
                    draw_corner_rect(frame, x1, y1, x2, y2, color, thickness=3, corner_len=16)

                    # Label text
                    conf_pct = int(conf * 100)
                    label_str = f" {info['display_name']} ({conf_pct}%) "
                    (tw, th), baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)

                    # Label background pill
                    ly2 = max(y1 - 4, th + 8)
                    ly1 = ly2 - th - 6
                    lx1 = x1
                    lx2 = x1 + tw

                    # Dark background behind label
                    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), COLOR_BG_DARK, -1)
                    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, 1, cv2.LINE_AA)
                    cv2.putText(frame, label_str, (lx1, ly2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw HUD bar
            draw_hud(frame, fps_smooth, counts, conf_thresh)

            # Bottom instruction tip
            h = frame.shape[0]
            tip_text = "Press [Q] to Exit | [S] Screenshot | [+] [-] Adjust Sensitivity"
            cv2.putText(frame, tip_text, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 190), 1, cv2.LINE_AA)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('s'):
                snap_name = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(snap_name, frame)
                print(f"[Saved] Snapshot saved as '{snap_name}'")
            elif key == ord('+') or key == ord('='):
                conf_thresh = min(0.90, conf_thresh + 0.05)
                print(f"[Setting] Confidence increased to: {conf_thresh:.2f}")
            elif key == ord('-') or key == ord('_'):
                conf_thresh = max(0.10, conf_thresh - 0.05)
                print(f"[Setting] Confidence decreased to: {conf_thresh:.2f}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n[Info] Camera released. Stream closed successfully.")

if __name__ == "__main__":
    main()
