import numpy as np
import cv2
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_pth", type=str, required=True,
                        help="Path to mask image (png/jpg)")
    args = parser.parse_args()

    mask_path = args.mask_pth

    # Load mask
    mask_img = cv2.imread(mask_path, 0)  # grayscale
    if mask_img is None:
        raise FileNotFoundError(f"Could not load mask image from: {mask_path}")

    mask_img = mask_img / 255.0  # Convert to 0~1 float

    # Save .npy next to mask_path
    base, _ = os.path.splitext(mask_path)
    mask_npy_path = base + ".npy"
    np.save(mask_npy_path, mask_img.astype(np.float32))

    print("Saved mask:", mask_npy_path)

if __name__ == "__main__":
    main()
