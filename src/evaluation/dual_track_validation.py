import cv2
import numpy as np

def synthetic_validation(image, known_angle=5.0, known_scale=1.0, known_tx=0, known_ty=0):
    """
    Ek REAL image lo, usme EK KNOWN transform apply karke ek fake "pair" banao.
    Phir apna poora pipeline (matching + RANSAC) use karke check karo ki
    recovered transform, known transform se kitna match karta hai.
    
    Ye ek non-circular ground truth deta hai — kyunki hume PEHLE SE pata hai
    sahi answer kya hona chahiye.
    
    Args:
        image: source image (numpy array)
        known_angle: rotation degrees jo hum apply karenge
        known_scale: scale factor
        known_tx, known_ty: translation (pixels)
    
    Returns:
        dict with validation results
    """
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'matching'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'filtering'))
    from classical_sift import match_sift
    from ransac import filter_ransac

    rows, cols = image.shape

    # KNOWN transform banate hain — ye hamara "ground truth" hai
    known_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), known_angle, known_scale)
    known_M[0, 2] += known_tx
    known_M[1, 2] += known_ty

    # Fake "pair" banate hain apply karke
    transformed_image = cv2.warpAffine(image, known_M, (cols, rows))

    # Ab pipeline chalate hain jaise real data ho
    matches = match_sift(image, transformed_image)
    if len(matches) < 4:
        return {"success": False, "reason": "Not enough matches found"}

    inliers, recovered_H = filter_ransac(matches)
    if recovered_H is None:
        return {"success": False, "reason": "Homography computation failed"}

    # Known transform ko 3x3 homography format mein convert karo comparison ke liye
    known_H = np.vstack([known_M, [0, 0, 1]])

    # Difference calculate karo — kitna recovered transform, known se alag hai
    diff_matrix = recovered_H - known_H
    max_diff = np.max(np.abs(diff_matrix))
    mean_diff = np.mean(np.abs(diff_matrix))

    # Ek threshold decide karo — chhota diff matlab pipeline sahi kaam kar raha hai
    passed = max_diff < 0.5  # tolerance, tune kar sakte ho real data pe

    return {
        "success": True,
        "passed": passed,
        "known_homography": known_H,
        "recovered_homography": recovered_H,
        "max_difference": max_diff,
        "mean_difference": mean_diff,
        "num_matches": len(matches),
        "num_inliers": len(inliers),
    }


if __name__ == "__main__":
    import rasterio

    with rasterio.open("data/raw/test_dummy.tif") as src:
        img = src.read(1)

    print("Running synthetic validation (known 5-degree rotation)...\n")
    result = synthetic_validation(img, known_angle=5.0)

    if result["success"]:
        print(f"✅ Matches found: {result['num_matches']}, Inliers: {result['num_inliers']}")
        print(f"\nKnown Homography:\n{result['known_homography']}")
        print(f"\nRecovered Homography:\n{result['recovered_homography']}")
        print(f"\nMax difference: {result['max_difference']:.4f}")
        print(f"Mean difference: {result['mean_difference']:.4f}")
        print(f"\n{'✅ PASSED' if result['passed'] else '❌ FAILED'} — pipeline {'correctly recovers' if result['passed'] else 'does NOT correctly recover'} known transform")
    else:
        print(f"❌ Validation failed: {result['reason']}")