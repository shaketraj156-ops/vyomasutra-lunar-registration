import cv2
import numpy as np


def match_sift(image1, image2, ratio_thresh=0.75):
    """
    SIFT features detect karke dono images ke beech match karta hai.

    Args:
        image1: source image (numpy array, grayscale)
        image2: reference image (numpy array, grayscale)
        ratio_thresh: Lowe's ratio test threshold (0.75 standard hai)

    Returns:
        List[(x1, y1, x2, y2, confidence)] — SHARED SCHEMA with Gaurav's LightGlue output.
        Agar input invalid hai ya koi match nahi mila, [] return karta hai (crash nahi karta).
    """
    # ---- NAYA: basic input validation, real data mein None/empty aa sakta hai ----
    if image1 is None or image2 is None:
        print("⚠️  match_sift: ek ya dono images None hain")
        return []

    if not isinstance(image1, np.ndarray) or not isinstance(image2, np.ndarray):
        print("⚠️  match_sift: images numpy array nahi hain")
        return []

    if image1.size == 0 or image2.size == 0:
        print("⚠️  match_sift: ek ya dono images empty hain (size 0)")
        return []

    if image1.ndim != 2 or image2.ndim != 2:
        print(f"⚠️  match_sift: grayscale (2D) images expected the, mila image1.ndim={image1.ndim}, image2.ndim={image2.ndim}")
        return []

    # ---- NAYA: ratio_thresh sanity bounds ----
    if not (0.0 < ratio_thresh <= 1.0):
        print(f"⚠️  match_sift: ratio_thresh={ratio_thresh} valid range (0, 1] se bahar hai, 0.75 use kar rahe hain")
        ratio_thresh = 0.75

    # Ensure 8-bit grayscale (SIFT requirement)
    try:
        if image1.dtype != np.uint8:
            image1 = cv2.normalize(image1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        if image2.dtype != np.uint8:
            image2 = cv2.normalize(image2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    except cv2.error as e:
        # ---- NAYA: normalize NaN/inf pe fail ho sakta hai (real sensor data mein possible) ----
        print(f"⚠️  match_sift: normalization fail hui (NaN/inf values ho sakte hain): {e}")
        return []

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        print("⚠️  Koi keypoints nahi mile — image mein enough texture nahi hai")
        return []

    # ---- NAYA: BFMatcher knnMatch k=2 ke liye kam se kam 2 descriptors chahiye ----
    if len(des2) < 2:
        print("⚠️  Reference image mein sirf 1 keypoint hai — knnMatch (k=2) ke liye kaafi nahi")
        return []

    bf = cv2.BFMatcher()
    raw_matches = bf.knnMatch(des1, des2, k=2)

    matches = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            x1, y1 = kp1[m.queryIdx].pt
            x2, y2 = kp2[m.trainIdx].pt
            confidence = max(0.0, 1.0 - (m.distance / 500.0))
            matches.append((x1, y1, x2, y2, confidence))

    return matches


if __name__ == "__main__":
    import rasterio

    with rasterio.open("data/raw/test_dummy.tif") as src:
        img1 = src.read(1)

    rows, cols = img1.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle=5, scale=1.0)
    img2 = cv2.warpAffine(img1, M, (cols, rows))

    print(f"Image 1 shape: {img1.shape}")
    print(f"Image 2 shape: {img2.shape}")

    matches = match_sift(img1, img2)

    print(f"\n✅ {len(matches)} matches mile")
    if matches:
        print(f"Sample match: {matches[0]}")
        print(f"Schema check: {type(matches[0])}, length={len(matches[0])}")