"""
metrics.py
Evaluation metrics for lunar image registration:
- Inlier Ratio (%)
- Geometric Reprojection RMSE (pixels)
- Structural Similarity Index (SSIM)
- Independent Ground Truth RMSE
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def compute_reprojection_rmse(inlier_matches: List, homography_matrix: np.ndarray) -> float:
    """
    Computes Root Mean Square Error (RMSE) in pixels between the projected source inliers
    and the actual reference points under the estimated Homography.
    
    Args:
        inlier_matches: List of inlier tuples [(x1, y1, x2, y2, ...)]
        homography_matrix: 3x3 homography matrix H
        
    Returns:
        float: Reprojection RMSE in pixels (sub-pixel if < 1.0)
    """
    if not inlier_matches or homography_matrix is None:
        return 0.0

    src_pts = np.array([[m[0], m[1]] for m in inlier_matches], dtype=np.float64).reshape(-1, 1, 2)
    dst_pts = np.array([[m[2], m[3]] for m in inlier_matches], dtype=np.float64).reshape(-1, 1, 2)

    try:
        # Project source points through H
        projected_pts = cv2.perspectiveTransform(src_pts, homography_matrix)
        diff = projected_pts - dst_pts
        squared_errors = np.sum(diff**2, axis=-1)  # shape (N, 1)
        rmse = float(np.sqrt(np.mean(squared_errors)))
        return rmse
    except Exception as e:
        print(f"⚠️ Reprojection RMSE computation error: {e}")
        return 0.0


def compute_inlier_ratio(total_matches: int, inlier_matches: List) -> float:
    """Computes percentage of geometrically consistent matches preserved by RANSAC."""
    if total_matches == 0:
        return 0.0
    return len(inlier_matches) / total_matches


def compute_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM) between registered and reference image.
    1.0 = identical, 0.0 = completely uncorrelated.
    """
    from skimage.metrics import structural_similarity as ssim

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    h, w = min(h1, h2), min(w1, w2)

    img1 = image1[:h, :w]
    img2 = image2[:h, :w]

    if img1.dtype != np.uint8:
        img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    try:
        score, _ = ssim(img1, img2, full=True)
        return float(score)
    except Exception as e:
        print(f"⚠️ SSIM calculation failed: {e}")
        return 0.0


def compute_rmse(pred_pts, true_pts) -> float:
    """Classical RMSE against independent control points."""
    pred = np.asarray(pred_pts, dtype=np.float64)
    true = np.asarray(true_pts, dtype=np.float64)
    diff = pred - true
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=-1))))