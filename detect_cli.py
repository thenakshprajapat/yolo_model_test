"""
Object Detection CLI Tool
Usage:
    python detect_cli.py --source static/samples/street_traffic.jpg
    python detect_cli.py --source static/samples/living_room.jpg --conf 0.3 --save
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 Object Detection from CLI")
    parser.add_argument("--source", type=str, default="static/samples/street_traffic.jpg", help="Path to image, video, directory, or webcam index (e.g. 0)")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Path to YOLO model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0.0 - 1.0)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (0.0 - 1.0)")
    parser.add_argument("--save", action="store_true", help="Save annotated detection results to runs/detect")
    parser.add_argument("--show", action="store_true", help="Display results in an OpenCV window")
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists() and not args.source.isdigit():
        print(f"[Error] Source path '{args.source}' does not exist.")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f"  YOLOv8 Object Detection CLI")
    print(f"  Weights   : {args.weights}")
    print(f"  Source    : {args.source}")
    print(f"  Confidence: {args.conf}")
    print(f"  IoU       : {args.iou}")
    print(f"=======================================================\n")

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save=args.save,
        show=args.show
    )

    for i, r in enumerate(results):
        print(f"--- Results for frame/image {i+1} ---")
        boxes = r.boxes
        if len(boxes) == 0:
            print("  No objects detected above confidence threshold.")
            continue

        print(f"  Found {len(boxes)} objects:")
        print(f"  {'Class':<18} | {'Confidence':<12} | {'Bounding Box (x1, y1, x2, y2)'}")
        print(f"  {'-'*18}-+-{'-'*12}-+-{'-'*35}")

        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = r.names[cls_id]
            conf = float(box.conf[0])
            coords = [round(float(c), 1) for c in box.xyxy[0]]
            print(f"  {cls_name:<18} | {conf*100:>6.1f}%       | {coords}")

        speed = r.speed
        print(f"\n  Inference speed: Preprocess: {speed['preprocess']:.1f}ms, Inference: {speed['inference']:.1f}ms, Postprocess: {speed['postprocess']:.1f}ms\n")

    if args.save:
        print(f"Saved outputs to 'runs/detect/predict'.")

if __name__ == "__main__":
    main()
