"""Candidate cutouts: masked RGB, context crop with neighbours, mask PNG and (gated) raw depth crop.

The depth crop reuses the RGB crop rectangle only under verified registration. There is no separate
depth-frame mapping in this slice, so without registration the depth crop is omitted with a reason.
"""
from . import rgbd

CUTOUT_VERSION = "ctx-v1"


def cutout_settings(margin_px=24):
    return {"cutout_version": CUTOUT_VERSION, "margin_px": int(margin_px)}


def make_cutouts(rgb, depth, mask_record, settings, registration_verified):
    import numpy as np
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = mask_record["bbox"]
    m = settings["margin_px"]
    cx0, cy0, cx1, cy1 = max(0, x0 - m), max(0, y0 - m), min(width, x1 + m), min(height, y1 + m)
    mask = mask_record["mask"]
    masked = np.zeros_like(rgb)
    masked[mask] = rgb[mask]
    out = {
        "cutout_png": rgbd.encode_rgb_png(masked[y0:y1, x0:x1]),
        "context_png": rgbd.encode_rgb_png(rgb[cy0:cy1, cx0:cx1]),
        "mask_png": rgbd.encode_mask_png(mask),
        "crop_transform": {"frame": "rgb_pixels", "x0": int(x0), "y0": int(y0), "width": int(x1 - x0),
                           "height": int(y1 - y0), "scale": 1},
        "context_transform": {"frame": "rgb_pixels", "x0": int(cx0), "y0": int(cy0), "width": int(cx1 - cx0),
                              "height": int(cy1 - cy0), "scale": 1},
        "cutout_version": settings["cutout_version"],
    }
    if registration_verified and depth.shape[:2] == rgb.shape[:2]:
        out["depth_crop_png"] = rgbd.encode_depth16_png(depth[y0:y1, x0:x1])
        out["depth_crop_reason"] = None
    else:
        out["depth_crop_png"] = None
        out["depth_crop_reason"] = "registration_not_verified"
    return out
