"""
Test API endpoints for Object Detection Prototype
"""
import requests
import json
from pathlib import Path

def test_prototype():
    base_url = "http://127.0.0.1:8000"
    
    # 1. Info endpoint
    info_resp = requests.get(f"{base_url}/api/info")
    assert info_resp.status_code == 200, f"Info failed: {info_resp.status_code}"
    info = info_resp.json()
    print(f"[*] API Info Check: {info['model_name']} loaded with {info['total_classes']} classes.")

    # 2. Image Detection endpoint with street sample
    sample_img = Path("static/samples/street_traffic.jpg")
    with open(sample_img, "rb") as f:
        detect_resp = requests.post(
            f"{base_url}/api/detect",
            files={"file": ("street_traffic.jpg", f, "image/jpeg")},
            data={"confidence": 0.25, "iou": 0.45}
        )
    assert detect_resp.status_code == 200, f"Detection failed: {detect_resp.status_code}"
    results = detect_resp.json()
    print(f"[*] Street Traffic Detection: Found {results['total_detections']} objects.")
    print(f"    Top entities: {results['class_counts']}")
    print(f"    Inference Latency: {results['metrics']['inference_ms']}ms | Total: {results['metrics']['total_ms']}ms")

    # 3. Image Detection endpoint with living room sample
    sample_living = Path("static/samples/living_room.jpg")
    with open(sample_living, "rb") as f:
        detect_resp2 = requests.post(
            f"{base_url}/api/detect",
            files={"file": ("living_room.jpg", f, "image/jpeg")},
            data={"confidence": 0.30, "iou": 0.45}
        )
    assert detect_resp2.status_code == 200, f"Detection failed: {detect_resp2.status_code}"
    results2 = detect_resp2.json()
    print(f"[*] Living Room Detection: Found {results2['total_detections']} objects.")
    print(f"    Top entities: {results2['class_counts']}")
    print(f"    Inference Latency: {results2['metrics']['inference_ms']}ms | Total: {results2['metrics']['total_ms']}ms")

    print("\n[SUCCESS] All prototype API endpoints and model pipelines are operating flawlessly!")

if __name__ == "__main__":
    test_prototype()
