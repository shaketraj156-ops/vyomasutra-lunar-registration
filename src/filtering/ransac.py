import cv2
import numpy as np
import sys
import os

def filter_ransac(matches, ransac_thresh=3.0):
    """
    RANSAC use karke bad/outlier matches filter karta hai.
    Sirf geometrically-consistent matches (inliers) rakhta hai.
    """
    if len(matches) < 4:
        print("⚠️  RANSAC ke liye kam se kam 4 matches chahiye")
        return [], None

    src_pts = np.float32([(m[0], m[1]) for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([(m[2], m[3]) for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

    if H is None:
        print("⚠️  Homography compute nahi ho payi")
        return [], None

    inlier_matches = [m for i, m in enumerate(matches) if mask[i]]

    return inlier_matches, H


if __name__ == "__main__":
    # Matching folder ka path add karo taaki classical_sift import ho sake
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'matching'))
    from classical_sift import match_sift
    import rasterio

    with rasterio.open("data/raw/test_dummy.tif") as src:
        img1 = src.read(1)

    rows, cols = img1.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle=5, scale=1.0)
    img2 = cv2.warpAffine(img1, M, (cols, rows))

    matches = match_sift(img1, img2)
    print(f"RANSAC se pehle: {len(matches)} matches")

    inliers, H = filter_ransac(matches)
    print(f"RANSAC ke baad: {len(inliers)} inlier matches")

    if H is not None:
        print(f"\n✅ Homography matrix mil gaya:")
        print(H)
    else:
        print("❌ Homography nahi mila")