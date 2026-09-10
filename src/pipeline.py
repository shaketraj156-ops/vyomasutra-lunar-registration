"""
pipeline.py
Central Execution Engine for Lunar Image Registration.
Orchestrates: Preprocessing -> Multi-modal Matching -> RANSAC Outlier Rejection ->
              Sub-Pixel Refinement -> Geometric Warping -> Evaluation Metrics.
"""

import os
import sys
import yaml
import time
import traceback
import cv2
import numpy as np
import rasterio
from typing import Dict, Any, Optional, Tuple

# Path resolution for internal imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
for sub in ['matching', 'filtering', 'refinement', 'warp', 'evaluation', 'preprocessing', 'acquisition']:
    p = os.path.join(CURRENT_DIR, sub)
    if p not in sys.path:
        sys.path.append(p)

from clahe import apply_clahe
from classical_sift import match_sift
from deep_lightglue import match_arrays as lightglue_match_arrays, LIGHTGLUE_AVAILABLE
from dual_matcher import combine_matches_arrays, strip_source_tag
from ransac import filter_ransac
from subpixel import refine_subpixel
from distribution_score import weighted_distribution_score
from transform import warp_image, create_checkerboard_overlay, export_geotiff
from metrics import compute_inlier_ratio, compute_ssim, compute_reprojection_rmse
from limitations_report import generate_limitations_report


def load_sensor_config(config_path: Optional[str] = None) -> Dict:
    """Loads sensors.yaml config file."""
    if config_path is None:
        config_path = os.path.join(CURRENT_DIR, "config", "sensors.yaml")
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[WARNING] Warning loading sensors.yaml: {e}")
    return {}


def run_registration_pipeline(
    source_img: np.ndarray,
    reference_img: np.ndarray,
    sensor_pair_key: str = "OHRC_LROC",
    matcher_type: str = "dual",  # "dual", "lightglue", "sift"
    apply_clahe_flag: bool = True,
    ransac_threshold: float = 3.0,
    enable_subpixel: bool = True,
    ref_crs = None,
    ref_transform = None,
    latitude: Optional[float] = None
) -> Dict[str, Any]:
    """
    End-to-end registration pipeline execution.
    """
    start_time = time.time()
    result = {
        "success": False,
        "error_message": None,
        "source_processed": None,
        "reference_processed": None,
        "warped_image": None,
        "checkerboard_image": None,
        "matches": [],
        "inliers": [],
        "homography": None,
        "metrics": {
            "total_matches": 0,
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "reprojection_rmse": 0.0,
            "distribution_score": 0.0,
            "ssim_score": 0.0,
        },
        "limitations": {},
        "elapsed_time_s": 0.0
    }

    try:
        # 1. Validation
        if source_img is None or reference_img is None:
            result["error_message"] = "Source or Reference image is None."
            return result

        if source_img.size == 0 or reference_img.size == 0:
            result["error_message"] = "Source or Reference image is empty (0 size)."
            return result

        # 2. Preprocessing & Bit-depth Normalization
        cfg = load_sensor_config()
        clip_limit = 2.0
        tile_size = (8, 8)
        sensor_name = sensor_pair_key.split("_")[0] if "_" in sensor_pair_key else sensor_pair_key
        if "sensors" in cfg and sensor_name in cfg["sensors"]:
            sensor_meta = cfg["sensors"][sensor_name].get("preprocessing", {})
            clip_limit = sensor_meta.get("clahe_clip_limit", 2.0)
            tile_size = tuple(sensor_meta.get("clahe_tile_grid_size", [8, 8]))

        if apply_clahe_flag:
            img1_proc = apply_clahe(source_img, clip_limit=clip_limit, tile_grid_size=tile_size)
            img2_proc = apply_clahe(reference_img, clip_limit=clip_limit, tile_grid_size=tile_size)
        else:
            img1_proc = source_img
            img2_proc = reference_img

        if img1_proc.dtype != np.uint8:
            img1_proc = cv2.normalize(img1_proc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img2_proc.dtype != np.uint8:
            img2_proc = cv2.normalize(img2_proc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        result["source_processed"] = img1_proc
        result["reference_processed"] = img2_proc
        print("[CHECKPOINT] 1. Preprocessing complete")

        # 3. Matching
        raw_matches = []
        if matcher_type == "sift":
            sift_pts = match_sift(img1_proc, img2_proc)
            raw_matches = [(*m, "sift") for m in sift_pts]
        elif matcher_type == "lightglue":
            if LIGHTGLUE_AVAILABLE:
                lg_pts = lightglue_match_arrays(img1_proc, img2_proc)
                raw_matches = [(*m, "lightglue") for m in lg_pts]
            else:
                print("[WARNING] LightGlue unavailable, falling back to SIFT")
                sift_pts = match_sift(img1_proc, img2_proc)
                raw_matches = [(*m, "sift") for m in sift_pts]
        else:  # "dual"
            raw_matches = combine_matches_arrays(img1_proc, img2_proc, run_sift=True, run_lightglue=LIGHTGLUE_AVAILABLE)

        result["matches"] = raw_matches
        result["metrics"]["total_matches"] = len(raw_matches)

        print(f"[CHECKPOINT] 2. Matching complete ({len(raw_matches)} matches)")

        if len(raw_matches) < 4:
            result["error_message"] = f"Insufficient matches ({len(raw_matches)} found). At least 4 required for geometric alignment."
            result["elapsed_time_s"] = time.time() - start_time
            return result

        # 4. Outlier Rejection via RANSAC
        clean_matches = strip_source_tag(raw_matches)
        inliers, H = filter_ransac(clean_matches, ransac_thresh=ransac_threshold)
        print(f"[CHECKPOINT] 3. RANSAC complete ({len(inliers)} inliers)")

        if H is None or len(inliers) < 4:
            result["error_message"] = "Geometric alignment failed: Homography matrix could not be estimated."
            result["elapsed_time_s"] = time.time() - start_time
            return result

        # 5. Sub-pixel Refinement
        final_inliers = inliers
        if enable_subpixel and len(inliers) >= 4:
            refined_inliers = refine_subpixel(img1_proc, inliers, image2=img2_proc)
            # Re-estimate homography with refined sub-pixel inliers
            refined_clean = strip_source_tag(refined_inliers) if len(refined_inliers[0]) > 5 else refined_inliers
            _, H_refined = filter_ransac(refined_clean, ransac_thresh=ransac_threshold)
            if H_refined is not None:
                H = H_refined
                final_inliers = refined_inliers
        print(f"[CHECKPOINT] 4. Subpixel complete ({len(final_inliers)} refined inliers)")

        result["inliers"] = final_inliers
        result["homography"] = H

        # 6. Geometric Warping onto Reference Canvas
        ref_h, ref_w = img2_proc.shape[:2]
        warped = warp_image(img1_proc, H, (ref_w, ref_h))
        if warped is None:
            result["error_message"] = "Image perspective warping failed."
            result["elapsed_time_s"] = time.time() - start_time
            return result

        result["warped_image"] = warped
        print("[CHECKPOINT] 5. Warping complete")

        # 7. Checkerboard Overlay
        result["checkerboard_image"] = create_checkerboard_overlay(warped, img2_proc, tile_size=48)
        print("[CHECKPOINT] 6. Checkerboard complete")

        # 8. Metrics Computation
        inlier_ratio = compute_inlier_ratio(len(raw_matches), final_inliers)
        reproj_rmse = compute_reprojection_rmse(final_inliers, H)
        dist_score = weighted_distribution_score(final_inliers, img1_proc.shape)
        ssim_val = compute_ssim(warped, img2_proc)
        print(f"[CHECKPOINT] 7. Metrics complete (RMSE={reproj_rmse:.4f}, SSIM={ssim_val:.4f})")

        result["metrics"]["inlier_count"] = len(final_inliers)
        result["metrics"]["inlier_ratio"] = float(inlier_ratio)
        result["metrics"]["reprojection_rmse"] = float(reproj_rmse)
        result["metrics"]["distribution_score"] = float(dist_score)
        result["metrics"]["ssim_score"] = float(ssim_val)

        # 9. Limitations Report
        try:
            lim_rep = generate_limitations_report(
                img1_proc, final_inliers, inlier_ratio, ssim_val, latitude=latitude
            )
            result["limitations"] = lim_rep
        except Exception as e:
            print(f"[WARNING] Limitations report error: {e}")

        result["success"] = True
        result["elapsed_time_s"] = round(time.time() - start_time, 3)

    except Exception as e:
        err_str = traceback.format_exc()
        print(f"[ERROR] Pipeline exception: {err_str}")
        result["error_message"] = f"Pipeline execution failed: {str(e)}"
        result["elapsed_time_s"] = round(time.time() - start_time, 3)

    return result
