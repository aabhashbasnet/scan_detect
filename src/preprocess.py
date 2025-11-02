import cv2
import numpy as np
import os


def preprocess_image(img_path, size=512):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize while maintaining aspect ratio
    h, w, _ = img.shape
    scale = size / max(h, w)
    newh, neww = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (neww, newh))

    # Pad to square
    pad_h, pad_w = (size - newh) // 2, (size - neww) // 2
    img_padded = cv2.copyMakeBorder(
        img_resized,
        pad_h,
        size - newh - pad_h,
        pad_w,
        size - neww - pad_w,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    # Normalize
    img_norm = img_padded / 255.0
    return img_norm


def preprocess_mask(mask_path, size=512):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (size, size))
    mask = mask / 255.0  # binary mask
    mask = np.expand_dims(mask, axis=0)  # (1,H,W)
    return mask
