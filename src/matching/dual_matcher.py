"""
dual_matcher.py
Combines classical (SIFT) and deep learning (LightGlue) matching results
into a single unified list, using the shared schema:
    List[(x1, y1, x2, y2, confidence, source_tag)]
Supports both direct numpy arrays and file paths.
"""

import os
from typing import List, Tuple, Union
import numpy as np
import rasterio

from classical_sift import match_sift
from deep_lightglue import match_arrays as lightglue_match_arrays, match_images as lightglue_match_images


def load_image_array(path: str) -> np.ndarray:
    """Load image into numpy array via rasterio or fallback to OpenCV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    try:
        with rasterio.open(path) as src:
            band1 = src.read(1)
            return band1
    except Exception:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Could not read image from {path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img


def combine_matches_arrays(
    img1: np.ndarray,
    img2: np.ndarray,
    run_sift: bool = True,
    run_lightglue: bool = True
) -> List[Tuple[float, float, float, float, float, str]]:
    """
    Runs SIFT and/or LightGlue on in-memory numpy arrays and combines the matches.
    """
    combined = []
    sift_matches_count = 0
    lg_matches_count = 0

    if run_sift:
        try:
            sift_matches = match_sift(img1, img2)
            sift_matches_count = len(sift_matches)
            for m in sift_matches:
                combined.append((*m, "sift"))
        except Exception as e:
            print(f"⚠️ SIFT matching warning: {e}")

    if run_lightglue:
        try:
            lg_matches = lightglue_match_arrays(img1, img2)
            lg_matches_count = len(lg_matches)
            for m in lg_matches:
                combined.append((*m, "lightglue"))
        except Exception as e:
            print(f"⚠️ LightGlue matching warning: {e}")

    print(f"Dual Matcher: SIFT={sift_matches_count}, LightGlue={lg_matches_count}, Total={len(combined)}")
    return combined


def combine_matches(
    source_path: str,
    reference_path: str,
    run_sift: bool = True,
    run_lightglue: bool = True
) -> List[Tuple[float, float, float, float, float, str]]:
    """Path-based wrapper for combine_matches_arrays."""
    img1 = load_image_array(source_path)
    img2 = load_image_array(reference_path)
    return combine_matches_arrays(img1, img2, run_sift=run_sift, run_lightglue=run_lightglue)


def strip_source_tag(tagged_matches: List) -> List[Tuple[float, float, float, float, float]]:
    """Strips the 'sift'/'lightglue' tag to return List[(x1, y1, x2, y2, confidence)]."""
    return [m[:5] for m in tagged_matches]