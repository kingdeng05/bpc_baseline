import numpy as np


def convert_t_to_site(t, K, bbox, s_zoom=256):
    """
    Args:
        t: [tx, ty, tz] 3D translation vector (GT) w.r.t. camera (meters)
        K: Camera intrinsics matrix (3x3)
        bbox: [x, y, w, h] bounding box in original image pixels
        s_zoom: size of zoomed-in (cropped) input image (default: 256)
    Returns:
        t_SITE: [delta_x, delta_y, delta_z] scale-invariant translation GT
    """
    tx, ty, tz = t
    ox, oy, _ = K @ np.array([tx, ty, tz])
    ox /= tz
    oy /= tz

    bbox_x, bbox_y, bbox_w, bbox_h = bbox
    cx = bbox_x + bbox_w / 2
    cy = bbox_y + bbox_h / 2

    delta_x = (ox - cx) / bbox_w
    delta_y = (oy - cy) / bbox_h
    r = s_zoom / max(bbox_w, bbox_h)
    delta_z = tz / r

    return np.array([delta_x, delta_y, delta_z])

def convert_site_to_t(site_pred, K, bbox, s_zoom=256):
    """
    Converts predicted scale-invariant SITE translation (delta_x, delta_y, delta_z)
    back to original 3D translation (x, y, z) in camera coordinates.

    Args:
        t_SITE_pred: [delta_x, delta_y, delta_z] (model's prediction)
        K: Camera intrinsics matrix (3x3)
        bbox: [x, y, w, h] bounding box used for cropping in the original image
        s_zoom: size of zoomed-in crop input to the model (default: 256)

    Returns:
        np.array: [tx, ty, tz] 3D translation in camera coordinates
    """
    delta_x, delta_y, delta_z = site_pred 

    bbox_x, bbox_y, bbox_w, bbox_h = bbox
    cx = bbox_x + bbox_w / 2
    cy = bbox_y + bbox_h / 2

    # Compute scaling ratio used during SITE preparation
    r = s_zoom / max(bbox_w, bbox_h)

    # Recover original depth tz
    tz = delta_z * r

    # Recover original 2D projection (ox, oy)
    ox = delta_x * bbox_w + cx
    oy = delta_y * bbox_h + cy

    # Back-project to 3D using camera intrinsics
    pixel_homog = np.array([ox, oy, 1.0])
    K_inv = np.linalg.inv(K)
    cam_point = K_inv @ pixel_homog
    cam_point *= tz

    tx, ty, tz = cam_point

    return np.array([tx, ty, tz])