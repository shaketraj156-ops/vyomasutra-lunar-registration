import numpy as np
import json
import os
from datetime import datetime

def compute_texture_variance(image, patch_size=16):
    """
    Image ko chhote patches mein baant kar har patch ka variance nikalta hai.
    Low variance = flat/low-texture area (jaise maria zones) — matching mushkil hoga yahan.
    """
    h, w = image.shape
    variance_map = np.zeros((h // patch_size, w // patch_size))
    
    for i in range(0, h - patch_size, patch_size):
        for j in range(0, w - patch_size, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            variance_map[i // patch_size, j // patch_size] = np.var(patch)
    
    return variance_map


def flag_low_texture_zones(image, patch_size=16, threshold_percentile=20):
    """
    Kaunse regions low-texture hain (flat, matching ke liye risky), flag karta hai.
    """
    variance_map = compute_texture_variance(image, patch_size)
    threshold = np.percentile(variance_map, threshold_percentile)
    
    low_texture_ratio = np.sum(variance_map < threshold) / variance_map.size
    
    return {
        "low_texture_ratio": float(low_texture_ratio),
        "mean_variance": float(np.mean(variance_map)),
        "warning": bool(low_texture_ratio > 0.3)
    }


def flag_polar_region(latitude=None):
    """
    Agar image ka latitude polar region (>75 degrees) ke paas hai,
    lower-confidence flag karta hai — kyunki lighting extreme hoti hai wahan.
    """
    if latitude is None:
        return {"is_polar": None, "note": "Latitude metadata not provided", "warning": False}
    
    is_polar = bool(abs(latitude) > 75.0)
    return {
        "is_polar": is_polar,
        "latitude": latitude,
        "warning": is_polar
    }


def generate_limitations_report(image, matches, inlier_ratio, ssim_score, 
                                  latitude=None, output_path="outputs/reports/limitations_report.json"):
    """
    Poora limitations report banata hai — texture, polar region, aur match quality
    ke based pe honest warnings deta hai. Ye JSON format mein save hota hai.
    """
    texture_info = flag_low_texture_zones(image)
    polar_info = flag_polar_region(latitude)
    
    warnings = []
    if texture_info["warning"]:
        warnings.append(
            f"Low-texture zones detected ({texture_info['low_texture_ratio']:.1%} of image) — "
            "matching confidence may be reduced in flat/maria regions."
        )
    if polar_info.get("warning"):
        warnings.append(
            f"Image is near a polar region (latitude {polar_info['latitude']}°) — "
            "extreme lighting conditions may reduce registration accuracy."
        )
    if inlier_ratio < 0.5:
        warnings.append(
            f"Inlier ratio is low ({inlier_ratio:.1%}) — fewer than half of matches were geometrically "
            "consistent. Results should be treated with lower confidence."
        )
    if len(matches) < 10:
        warnings.append(
            f"Only {len(matches)} matches found — sparse matches increase risk of an unreliable "
            "homography estimate."
        )
    if not warnings:
        warnings.append("No significant limitations detected for this image pair.")

    report = {
        "generated_at": datetime.now().isoformat(),
        "texture_analysis": texture_info,
        "polar_region_analysis": polar_info,
        "match_quality": {
            "total_matches": int(len(matches)),
            "inlier_ratio": float(inlier_ratio),
            "ssim_score": float(ssim_score),
        },
        "warnings": warnings,
        "confidence_level": "HIGH" if not any([texture_info["warning"], polar_info["warning"], inlier_ratio < 0.5]) else "MODERATE" if inlier_ratio >= 0.5 else "LOW",
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'matching'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'filtering'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'warp'))
    from classical_sift import match_sift
    from ransac import filter_ransac
    from metrics import compute_inlier_ratio, compute_ssim
    from transform import warp_image
    import rasterio
    import cv2

    with rasterio.open("data/raw/test_dummy.tif") as src:
        img1 = src.read(1)

    rows, cols = img1.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle=5, scale=1.0)
    img2 = cv2.warpAffine(img1, M, (cols, rows))

    matches = match_sift(img1, img2)
    inliers, H = filter_ransac(matches)
    warped = warp_image(img1, H, (cols, rows))

    inlier_ratio = compute_inlier_ratio(len(matches), inliers)
    ssim_score = compute_ssim(warped, img2)

    report = generate_limitations_report(
        img1, matches, inlier_ratio, ssim_score,
        latitude=-10.0  # test coordinate, real data pe actual latitude use karo
    )

    print("✅ Limitations report generated:\n")
    print(json.dumps(report, indent=2))