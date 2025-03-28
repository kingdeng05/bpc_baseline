from tqdm import tqdm
from torch.utils.data import DataLoader

from bpc.utils.multi_cam_dataset import BOPMultiCamDataset, bop_collate_fn_multi_cam

scene_ids = [f"{i:06d}" for i in range(50)]
ds = BOPMultiCamDataset("datasets", scene_ids, cam_ids=["cam1", "cam2", "cam3"], target_obj_id=20, target_size=256, augment=False, split="train")
# train_loader = DataLoader(ds, 12, shuffle=True, num_workers=10, collate_fn=bop_collate_fn_multi_cam)
# for data in tqdm(train_loader):
#     pass