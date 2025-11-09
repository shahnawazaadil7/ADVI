import open3d as o3d
import numpy as np
import json
import os

def rot_y(deg):
    rad = np.deg2rad(deg)
    return np.array([
        [ np.cos(rad), 0, np.sin(rad)],
        [ 0,           1, 0          ],
        [-np.sin(rad), 0, np.cos(rad)]
    ])

CAMERA_INTRINSICS = {
    "CAM_FRONT":        np.array([[1200,    0, 800], [0, 1200, 450], [0, 0, 1]]),
    "CAM_FRONT_LEFT":   np.array([[1200,    0, 800], [0, 1200, 450], [0, 0, 1]]),
    "CAM_FRONT_RIGHT":  np.array([[1200,    0, 800], [0, 1200, 450], [0, 0, 1]]),
    "CAM_BACK":         np.array([[1200,    0, 800], [0, 1200, 450], [0, 0, 1]]),
    "CAM_BACK_LEFT":    np.array([[1200,    0, 800], [0, 1200, 450], [0, 0, 1]]),
    "CAM_BACK_RIGHT":   np.array([[1200,    0, 800], [0, 1200, 450], [0, 0, 1]]),
}

def rot_z(deg):
    rad = np.deg2rad(deg)
    return np.array([
        [np.cos(rad), -np.sin(rad), 0],
        [np.sin(rad),  np.cos(rad), 0],
        [0,            0,           1]
    ])

CAMERA_EXTRINSICS = {
    "CAM_FRONT":        np.vstack([np.hstack([rot_y(0),   np.array([[3.5], [0], [1.5]])]), [0,0,0,1]]),
    "CAM_FRONT_LEFT":   np.vstack([np.hstack([rot_y(45),  np.array([[3.3], [0.8], [1.5]])]), [0,0,0,1]]),
    "CAM_FRONT_RIGHT":  np.vstack([np.hstack([rot_y(-45), np.array([[3.3], [-0.8], [1.5]])]), [0,0,0,1]]),
    "CAM_BACK":         np.vstack([np.hstack([rot_y(180), np.array([[-1.0], [0], [1.5]])]), [0,0,0,1]]),
    "CAM_BACK_LEFT":    np.vstack([np.hstack([rot_y(135), np.array([[-1.0], [0.8], [1.5]])]), [0,0,0,1]]),
    "CAM_BACK_RIGHT":   np.vstack([np.hstack([rot_y(-135),np.array([[-1.0], [-0.8], [1.5]])]), [0,0,0,1]]),
}

RADAR_EXTRINSICS = [
    np.vstack([np.hstack([rot_z(90),   np.array([[3.5], [0], [0.5]])]), [0, 0, 0, 1]]),
    np.vstack([np.hstack([rot_z(135),  np.array([[3.3], [0.8], [0.5]])]), [0, 0, 0, 1]]),
    np.vstack([np.hstack([rot_z(45),   np.array([[3.3], [-0.8], [0.5]])]), [0, 0, 0, 1]]),
    np.vstack([np.hstack([rot_z(225),  np.array([[-1.0], [0.8], [0.5]])]), [0, 0, 0, 1]]),
    np.vstack([np.hstack([rot_z(-45),  np.array([[-1.0], [-0.8], [0.5]])]), [0, 0, 0, 1]])
]

def load_lidar(lidar_path):
    import numpy as np
    import os
    if not os.path.exists(lidar_path):
        print(f"[WARN] LiDAR file not found: {lidar_path}")
        return np.empty((0, 3))
    points = np.fromfile(lidar_path, dtype=np.float32)
    if points.size % 5 == 0:
        points = points.reshape(-1, 5)[:, :3]
    elif points.size % 4 == 0:
        points = points.reshape(-1, 4)[:, :3]
    else:
        points = points.reshape(-1, 3)
    print(f"[INFO] Loaded LiDAR: {lidar_path}, points: {points.shape[0]}")
    return points

def load_radar(radar_path):
    import open3d as o3d
    import numpy as np
    import os
    if not os.path.exists(radar_path):
        print(f"[WARN] Radar file not found: {radar_path}")
        return np.empty((0, 4))
    pcd = o3d.io.read_point_cloud(radar_path)
    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        print(f"[WARN] Radar file empty: {radar_path}")
    vx = np.random.uniform(0, 20, size=(pts.shape[0], 1))
    print(f"[INFO] Loaded Radar: {radar_path}, points: {pts.shape[0]}")
    return np.hstack([pts, vx])

def project_lidar_to_image(lidar_points, cam_name):
    if lidar_points.shape[0] == 0:
        return np.empty((0,2)), np.array([], dtype=bool)
    K = CAMERA_INTRINSICS[cam_name]
    T = CAMERA_EXTRINSICS[cam_name]
    pts_h = np.hstack([lidar_points, np.ones((lidar_points.shape[0], 1))])
    pts_cam = (np.linalg.inv(T) @ pts_h.T).T[:, :3]
    mask = pts_cam[:,2] > 0.1
    pts_cam = pts_cam[mask]
    pts_img = (K @ pts_cam.T).T
    pts_img = pts_img[:, :2] / pts_img[:, 2:3]
    return pts_img, mask

def project_radar_to_image(radar_points, cam_name, radar_idx):
    if radar_points.shape[0] == 0:
        return np.empty((0,2)), np.array([], dtype=bool)
    K = CAMERA_INTRINSICS[cam_name]
    T_cam = CAMERA_EXTRINSICS[cam_name]
    T_radar = RADAR_EXTRINSICS[radar_idx]
    # Transform radar points to homogeneous
    pts_h = np.hstack([radar_points[:, :3], np.ones((radar_points.shape[0], 1))])
    # Radar -> vehicle (LiDAR) frame
    pts_vehicle = (T_radar @ pts_h.T).T[:, :3]
    # Vehicle -> camera frame
    pts_cam = (np.linalg.inv(T_cam) @ np.hstack([pts_vehicle, np.ones((pts_vehicle.shape[0], 1))]).T).T[:, :3]
    mask = pts_cam[:,2] > 0.1
    pts_cam = pts_cam[mask]
    pts_img = (K @ pts_cam.T).T
    pts_img = pts_img[:, :2] / pts_img[:, 2:3]
    return pts_img, mask