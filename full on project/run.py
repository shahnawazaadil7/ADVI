import os
import cv2
import numpy as np
from src.detect_objects import detect_objects_yolo
from src.fuse_sensors import load_lidar, load_radar, project_lidar_to_image, project_radar_to_image
from src.risk_estimator import estimate_risk
from src.overlay_visuals import draw_overlay

CAM_NAMES = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
]
RADAR_NAMES = [
    "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"
]
DATA_ROOT = "samples"
LIDAR_DIR = os.path.join(DATA_ROOT, "LIDAR_TOP")

def find_closest_file(target_id, folder, ext):
    """
    Finds the file in 'folder' with extension 'ext' whose name is most similar to 'target_id'.
    """
    files = [f for f in os.listdir(folder) if f.endswith(ext)]
    if not files:
        print(f"[WARN] No files with extension {ext} in {folder}")
        return None
    # Try to match by numeric timestamp if possible
    import re
    def extract_digits(s):
        return ''.join(re.findall(r'\d+', s))
    t_id = extract_digits(target_id)
    def score(f):
        f_id = extract_digits(f)
        if t_id and f_id:
            return abs(int(t_id) - int(f_id))
        # fallback: string difference
        return sum(a != b for a, b in zip(f, target_id))
    files.sort(key=lambda f: score(f))
    chosen = files[0]
    print(f"[INFO] Matched {target_id} to {chosen} in {folder}")
    return os.path.join(folder, chosen)

for cam in CAM_NAMES:
    cam_dir = os.path.join(DATA_ROOT, cam)
    frame_names = sorted([f for f in os.listdir(cam_dir) if f.endswith('.jpg')])
    if not frame_names:
        print(f"[WARN] No images found in {cam_dir}")
        continue
    for frame in frame_names[:1]:  # Demo: just first frame per camera
        img_path = os.path.join(cam_dir, frame)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue

        # Find closest LiDAR file
        lidar_file = find_closest_file(frame, LIDAR_DIR, ".pcd.bin")
        lidar_points = load_lidar(lidar_file) if lidar_file else np.empty((0, 3))

        # Find closest Radar files for all radars
        radar_points = []
        for i, radar in enumerate(RADAR_NAMES):
            radar_folder = os.path.join(DATA_ROOT, radar)
            radar_file = find_closest_file(frame, radar_folder, ".pcd")
            radar_pts = load_radar(radar_file) if radar_file else np.empty((0, 4))
            radar_points.append(radar_pts)

        detections = detect_objects_yolo(img)
        print(f"[INFO] {len(detections)} detections in {img_path}")

        # Project LiDAR points to image
        pts_img, mask = project_lidar_to_image(lidar_points, cam)
        # Project Radar points to image (flatten all radars for demo)
        radar_img_pts = []
        radar_vels = []
        for i, radar_pts in enumerate(radar_points):
            if radar_pts.shape[0] == 0:
                continue
            pts_radar_img, mask_radar = project_radar_to_image(radar_pts, cam, radar_idx=i)
            radar_img_pts.append(pts_radar_img)
            radar_vels.append(radar_pts[:, 3])
        if radar_img_pts:
            radar_img_pts = np.vstack(radar_img_pts)
            radar_vels = np.hstack(radar_vels)
        else:
            radar_img_pts = np.empty((0,2))
            radar_vels = np.empty((0,))

        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            # LiDAR distance
            if pts_img.shape[0] > 0:
                in_box = (pts_img[:,0] >= x1) & (pts_img[:,0] <= x2) & (pts_img[:,1] >= y1) & (pts_img[:,1] <= y2)
                if np.any(in_box):
                    det['distance'] = float(np.median(lidar_points[mask][in_box,2]))
                    print(f"[INFO] LiDAR distance for {det['class']}: {det['distance']:.2f}m")
                else:
                    det['distance'] = 10.0
                    print(f"[WARN] No LiDAR points in bbox for {det['class']}")
            else:
                det['distance'] = 10.0
                print(f"[WARN] No LiDAR points projected for {det['class']}")

            # Radar velocity
            if radar_img_pts.shape[0] > 0:
                in_box_radar = (radar_img_pts[:,0] >= x1) & (radar_img_pts[:,0] <= x2) & (radar_img_pts[:,1] >= y1) & (radar_img_pts[:,1] <= y2)
                if np.any(in_box_radar):
                    det['velocity'] = float(np.median(radar_vels[in_box_radar]))
                    print(f"[INFO] Radar velocity for {det['class']}: {det['velocity']:.2f}m/s")
                else:
                    det['velocity'] = 0.0
                    print(f"[WARN] No Radar points in bbox for {det['class']}")
            else:
                det['velocity'] = 0.0
                print(f"[WARN] No Radar points projected for {det['class']}")

            det['risk'] = estimate_risk(det['distance'], det['velocity'])

        annotated = draw_overlay(img, detections)
        out_dir = f"results/annotated_images/{cam}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, frame.replace('.jpg', '_annotated.jpg'))
        cv2.imwrite(out_path, annotated)
        print(f"[INFO] Saved: {out_path}")