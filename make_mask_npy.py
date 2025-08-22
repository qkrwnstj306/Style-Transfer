import numpy as np
import cv2
import os

content_path = "./dataset/cnt/content_flower.jpg"
mask_path = "./dataset/mask/content_flower_mask.png"  

# Load mask.jpg
mask_img = cv2.imread(mask_path, 0)  # grayscale
if mask_img is None:
    raise FileNotFoundError(f"Could not load mask image from: {mask_path}")

mask_img = mask_img / 255.0  # Convert to 0~1 float

# Save as .npy
mask_npy_path = "./dataset/mask/content_flower_mask.npy"
np.save(mask_npy_path, mask_img.astype(np.float32))

print("Saved mask:", mask_npy_path)
