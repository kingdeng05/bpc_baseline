import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
import cv2
import glob
import json
import time
import trimesh
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, precision=3)
from scipy.spatial.transform import Rotation as R

# Local imports from your project:
from bpc.inference.utils.camera_utils import load_camera_params
from bpc.inference.process_pose import PoseEstimator, PoseEstimatorParams
from bpc.utils.data_utils import Capture, render_mask
import bpc.utils.data_utils as du
import importlib

def model_snapshot_test():
    # Example paths and scene setup
    scene_dir = "./datasets/val/000003/"
    models_dir = './datasets/models/'
    cam_ids = ["cam1", "cam2", "cam3"]
    image_id = 0
    obj_id = 11
    ply_file = os.path.join(models_dir, f"obj_{obj_id:06d}.ply")
    obj = trimesh.load(ply_file)

    # YOLO and pose model paths
    yolo_model_path = f'yolo/models/detection/obj_{obj_id}/yolo11-detection-obj_{obj_id}.pt'
    pose_model_path = f'ckpts_path/obj_{obj_id}/best_model.pth'

    # Configure the pose estimator
    pose_params = PoseEstimatorParams(
        yolo_model_path=yolo_model_path,
        pose_model_path=pose_model_path,
        yolo_conf_thresh=0.01,
        rotation_mode="euler"  # Using quaternion mode for the example.
    )
    pose_estimator = PoseEstimator(pose_params)

    # Perform multi-camera detection, matching, and rotation inference
    t_start = time.time()
    capture = Capture.from_dir(scene_dir, cam_ids, image_id, obj_id)
    detections = pose_estimator._detect(capture)
    pose_predictions = pose_estimator._match(capture, detections)
    pose_estimator._estimate_rotation(pose_predictions)
    inference_time = time.time() - t_start
    print(pose_predictions[0].pose)
    print("Inference took:", inference_time, "seconds")

# pose_predictions now contains the final poses for each matched object
# Evaluate these against ground-truth poses with your desired BOP metric.
def eval(obj_id):
    import csv

    # YOLO and pose model paths
    yolo_model_path = f'yolo/models/detection/obj_{obj_id}/yolo11-detection-obj_{obj_id}.pt'
    pose_model_path = f'ckpts_path/obj_{obj_id}/best_model.pth'

    # Configure the pose estimator
    pose_params = PoseEstimatorParams(
        yolo_model_path=yolo_model_path,
        pose_model_path=pose_model_path,
        yolo_conf_thresh=0.01,
        rotation_mode="euler"  # Using quaternion mode for the example.
    )
    pose_estimator = PoseEstimator(pose_params)

    csv_file = "./baseline_datasets-val.csv"
    cam_ids = ["cam1", "cam2", "cam3"]

    ds_path = "datasets/val"
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "scene_id", "im_id", "obj_id", "score",
            "R", "t", "time"
        ])
        for scene in os.listdir(ds_path):
            scene_path = os.path.join(ds_path, scene)
            with open(os.path.join(scene_path, f"scene_gt_{cam_ids[0]}.json")) as f:
                gt_info = json.load(f)
            for im_id in sorted(gt_info.keys()):
                for ins in gt_info[str(im_id)]:
                    obj_id_cur = ins["obj_id"]
                    if obj_id_cur == obj_id:
                        t_start = time.time()
                        capture = Capture.from_dir(scene_path, cam_ids, int(im_id), obj_id)
                        detections = pose_estimator._detect(capture)
                        pose_predictions = pose_estimator._match(capture, detections)
                        pose_estimator._estimate_rotation(pose_predictions)
                        inference_time = time.time() - t_start
                        for pred in pose_predictions:
                            row = [
                                int(scene),
                                int(im_id),
                                obj_id,
                                1.,
                                " ".join([str(e) for e in pred.pose[:3, :3].flatten().tolist()]),
                                " ".join([str(e) for e in pred.pose[:3, 3].tolist()]),
                                -1 
                            ]
                            writer.writerow(row)


if __name__ == "__main__":
    eval(11)
