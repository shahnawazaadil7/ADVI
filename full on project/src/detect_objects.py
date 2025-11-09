from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Downloaded automatically

def detect_objects_yolo(image):
    results = model(image)
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                'bbox': box.xyxy[0].cpu().numpy().tolist(),
                'class': model.names[int(box.cls[0])],
                'confidence': float(box.conf[0])
            })
    return detections