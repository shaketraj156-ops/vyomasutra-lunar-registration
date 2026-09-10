"""
subpixel.py
Sub-pixel keypoint refinement using OpenCV cornerSubPix.
Performs bidirectional refinement on both source and reference images.
"""

import cv2
import numpy as np
from typing import List, Tuple

Match = Tuple[float, float, float, float, float]


def refine_subpixel(
    image1: np.ndarray,
    matches: List[Match],
    image2: np.ndarray = None,
    win_size: Tuple[int, int] = (5, 5),
    zero_zone: Tuple[int, int] = (-1, -1),
    max_iter: int = 30,
    eps: float = 0.001
) -> List[Match]:
    """
    Refines keypoint coordinates on image1 (and optionally image2) to sub-pixel accuracy.
    
    Args:
        image1: Source grayscale image (2D numpy array)
        matches: List[(x1, y1, x2, y2, confidence, ...)]
        image2: Reference grayscale image (optional, if provided refines x2, y2 as well)
        win_size: Half of the side length of the search window
        zero_zone: Half of the size of the dead region in the middle
        max_iter: Maximum iterations
        eps: Desired accuracy
        
    Returns:
        Refined matches with floating sub-pixel coordinates.
    """
    if not matches:
        return []

    # Ensure 8-bit grayscale for OpenCV cornerSubPix
    if image1.dtype != np.uint8:
        img1_8u = cv2.normalize(image1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    else:
        img1_8u = image1.copy()

    # Prepare points1: shape (N, 1, 2)
    pts1 = np.array([[m[0], m[1]] for m in matches], dtype=np.float32).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)

    try:
        refined_pts1 = cv2.cornerSubPix(img1_8u, pts1, win_size, zero_zone, criteria)
    except cv2.error as e:
        print(f"⚠️ subpixel refinement failed on image1: {e}")
        refined_pts1 = pts1

    # If image2 is provided, refine reference coordinates as well
    if image2 is not None:
        if image2.dtype != np.uint8:
            img2_8u = cv2.normalize(image2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        else:
            img2_8u = image2.copy()

        pts2 = np.array([[m[2], m[3]] for m in matches], dtype=np.float32).reshape(-1, 1, 2)
        try:
            refined_pts2 = cv2.cornerSubPix(img2_8u, pts2, win_size, zero_zone, criteria)
        except cv2.error as e:
            print(f"⚠️ subpixel refinement failed on image2: {e}")
            refined_pts2 = pts2
    else:
        refined_pts2 = np.array([[m[2], m[3]] for m in matches], dtype=np.float32).reshape(-1, 1, 2)

    refined_matches = []
    for i, m in enumerate(matches):
        rx1, ry1 = refined_pts1[i][0]
        rx2, ry2 = refined_pts2[i][0]
        extra_tags = m[5:] if len(m) > 5 else ()
        refined_matches.append((float(rx1), float(ry1), float(rx2), float(ry2), float(m[4]), *extra_tags))

    return refined_matches