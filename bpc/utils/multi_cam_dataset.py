from torch.utils.data import Dataset
from collections import defaultdict
import os, json, random, cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from .data_utils import letterbox_preserving_aspect_ratio, matrix_to_euler_xyz, euler_to_6d, euler_to_quat 


def bop_collate_fn_multi_cam(batch):
    """
    Collates a batch of multi-view samples.
    
    Each sample is expected to be a tuple:
      (imgs, lbls, metas)
    where:
      - imgs is a tensor of shape [num_views, C, H, W]
      - lbls is a list of label dictionaries (one per view)
      - metas is a list of metadata dictionaries (one per view)
      
    The collate function returns:
      - imgs_batch: tensor of shape [B, num_views, C, H, W]
      - labels: list of length B, each element is a list of label dicts.
      - metas: list of length B, each element is a list of metadata dicts.
    """
    imgs_list, labels_list, metas_list = zip(*batch)
    # Stack images along the new batch dimension.
    imgs_batch = torch.stack(imgs_list, dim=0)
    return imgs_batch, list(labels_list), list(metas_list)


class BOPMultiCamDataset(Dataset):
    """
    A dataset that groups images from multiple cameras for a single object ID.
    For each sample (grouped by scene_id and im_id), it returns:
      - A list of image tensors (one per camera)
      - A list of label dictionaries (one per camera)
      - A list of metadata dictionaries (one per camera)
    """
    def __init__(self,
                 root_dir,
                 scene_ids,
                 cam_ids,
                 target_obj_id,
                 target_size=256,
                 augment=False,
                 split="train",
                 max_per_scene=None,
                 train_ratio=0.8,
                 seed=42):
        super().__init__()
        self.root_dir = root_dir
        self.scene_ids = scene_ids
        self.cam_ids = cam_ids
        self.obj_id = target_obj_id
        self.target_size = target_size
        self.augment = augment
        self.split = split.lower()  # "train" or "val"
        self.max_per_scene = max_per_scene
        self.train_ratio = train_ratio
        self.samples_grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list))) 
        random.seed(seed)

        train_pbr_path = os.path.join(root_dir, "train_pbr")
        for sid in scene_ids:
            scene_path = os.path.join(train_pbr_path, sid)
            scene_count = 0
            for cam_id in cam_ids:
                info_file = os.path.join(scene_path, f"scene_gt_info_{cam_id}.json")
                pose_file = os.path.join(scene_path, f"scene_gt_{cam_id}.json")
                cam_file  = os.path.join(scene_path, f"scene_camera_{cam_id}.json")
                rgb_dir   = os.path.join(scene_path, f"rgb_{cam_id}")
                if not all(os.path.exists(f) for f in [info_file, pose_file, cam_file, rgb_dir]):
                    continue
                with open(info_file, "r") as f1, open(pose_file, "r") as f2, open(cam_file, "r") as f3:
                    info_json = json.load(f1)
                    pose_json = json.load(f2)
                    cam_json  = json.load(f3)
                all_im_ids = sorted(info_json.keys(), key=lambda x: int(x))
                for im_id_s in all_im_ids:
                    im_id = int(im_id_s)
                    if im_id_s not in cam_json:
                        continue
                    K = np.array(cam_json[im_id_s]["cam_K"], dtype=np.float32).reshape(3, 3)
                    img_name = f"{im_id:06d}.jpg"
                    img_path = os.path.join(rgb_dir, img_name)
                    if not os.path.exists(img_path):
                        continue
                    # Loop through object instances in the image.
                    for inf, pos in zip(info_json[im_id_s], pose_json[im_id_s]):
                        if pos["obj_id"] != self.obj_id:
                            continue
                        x, y, w_, h_ = inf["bbox_visib"]
                        if w_ <= 0 or h_ <= 0:
                            continue
                        R_mat = np.array(pos["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
                        t = np.array(pos["cam_t_m2c"], dtype=np.float32).reshape(3, 1)
                        sample = {
                            "scene_id": sid,
                            "cam_id": cam_id,
                            "im_id": im_id,
                            "img_path": img_path,
                            "K": K,
                            "R": R_mat,
                            "t": t,
                            "bbox_visib": [x, y, w_, h_]
                        }
                        self.samples_grouped[sid][im_id][cam_id].append(sample) # will append the obj in
                scene_count += 1
                if self.max_per_scene is not None and scene_count >= self.max_per_scene:
                    break

            # check if all cam have the same # of objects
            for im_id in all_im_ids:
                obj_num = None
                for cam_id in cam_ids:
                    num = len(self.samples_grouped[sid][im_id][cam_id])
                    if obj_num is None:
                        obj_num = num
                    if num != obj_num:
                        raise ValueError(f"for cam {cam_id} in {im_id} isn't conforming!")

        # Now split each group into train and validation portions.
        self.samples = []
        for sid, im_dict in self.samples_grouped.items():
            for im_id, cam_dict in im_dict.items():
                choices = []
                # this enforces the same object in all cameras will be assigned to one split
                for cam_id, obj_lst in cam_dict.items():
                    n_total = len(obj_lst)
                    n_train = int(round(self.train_ratio * n_total))
                    choices = np.random.choice(n_total, n_train).tolist()
                    break
                if self.split != "train":
                    choices = set(n_total).difference(set(choices)) 
                for i in choices:
                    sample = {}
                    for cam_id, obj_lst in cam_dict.items():
                        print(f"{sid} {im_id} {cam_id} len: {len(obj_lst)}")
                        sample[cam_id] = obj_lst[i] 
                    self.samples.append(sample)

        print(f"[INFO] BOPMultiCamDataset(split={self.split}, augment={self.augment}): total groups={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Each sample is a group of camera views for the same (scene_id, im_id).
        group = self.samples[idx]
        imgs, label_dicts, metas = [], [], []
        for data in group:
            img_path = data["img_path"]
            bgr = cv2.imread(img_path)
            if bgr is None:
                raise IOError(f"Cannot read {img_path}")
            H_img, W_img = bgr.shape[:2]
            x, y, w, h = map(int, data["bbox_visib"])
            
            # (Optional) Apply augmentation to the bounding box if desired.
            if self.augment and self.split == "train":
                scale_factor = 1.0 + 0.2 * random.random()
                new_w = int(round(w * scale_factor))
                new_h = int(round(h * scale_factor))
                max_shift_x = int(0.1 * w)
                max_shift_y = int(0.1 * h)
                shift_x = random.randint(-max_shift_x, max_shift_x)
                shift_y = random.randint(-max_shift_y, max_shift_y)
                x0 = x - shift_x
                y0 = y - shift_y
                x0 = max(0, min(x0, W_img - 1))
                y0 = max(0, min(y0, H_img - 1))
                new_w = min(new_w, W_img - x0)
                new_h = min(new_h, H_img - y0)
            else:
                x0, y0 = x, y
                new_w, new_h = w, h

            crop = bgr[y0:y0+new_h, x0:x0+new_w]
            if crop.size == 0:
                raise RuntimeError("Empty crop => skip")
            letter_img, scale, dx, dy = letterbox_preserving_aspect_ratio(crop, target_size=self.target_size)
            # Compute rotation representations.
            Rx, Ry, Rz = matrix_to_euler_xyz(data["R"])
            euler_angles = np.array([Rx, Ry, Rz], dtype=np.float32)
            quat = euler_to_quat(euler_angles)
            rep6d = euler_to_6d(euler_angles)
            label_dict = {
                "euler": torch.from_numpy(euler_angles),
                "quat": torch.from_numpy(np.array(quat, dtype=np.float32)),
                "6d": torch.from_numpy(np.array(rep6d, dtype=np.float32))
            }
            # Preprocess image: convert to tensor, normalize, etc.
            letter_img_c = np.ascontiguousarray(letter_img, dtype=np.uint8)
            img_t = torch.from_numpy(letter_img_c).permute(2, 0, 1).float() / 255.0
            # (Optional) Color jitter if augmenting.
            if self.augment and self.split == "train":
                jitter_transform = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                img_pil = TF.to_pil_image(img_t)
                img_pil = jitter_transform(img_pil)
                img_t = TF.to_tensor(img_pil)
            # Normalize with ImageNet stats.
            img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
            meta = {
                "scene_id": data["scene_id"],
                "cam_id": data["cam_id"],
                "im_id": data["im_id"]
            }
            imgs.append(img_t)
            label_dicts.append(label_dict)
            metas.append(meta)
            
        # Stack images into a single tensor (e.g., [N, C, H, W] where N is the number of cameras)
        imgs_t = torch.stack(imgs, dim=0)
        return imgs_t, label_dicts, metas

