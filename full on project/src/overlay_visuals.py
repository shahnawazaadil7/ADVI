import cv2

def draw_overlay(image, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        color = (0,255,0)
        if det['risk'] == 'high':
            color = (0,0,255)
        elif det['risk'] == 'medium':
            color = (0,255,255)
        label = f"{det['class']} {det.get('distance', '?'):.1f}m {det.get('velocity', '?'):.1f}m/s"
        cv2.rectangle(image, (x1,y1), (x2,y2), color, 2)
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image