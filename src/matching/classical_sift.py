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
        x1,y1 = point in image1; x2,y2 = corresponding point in image2;
        confidence = 0 to 1 (higher = better match)
    """
    # Ensure 8-bit grayscale (SIFT requirement)
    if image1.dtype != np.uint8:
        image1 = cv2.normalize(image1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    if image2.dtype != np.uint8:
        image2 = cv2.normalize(image2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        print("⚠️  Koi keypoints nahi mile — image mein enough texture nahi hai")
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
            # confidence: distance ko 0-1 scale mein convert karo (chhota distance = zyada confidence)
            confidence = max(0.0, 1.0 - (m.distance / 500.0))
            matches.append((x1, y1, x2, y2, confidence))

    return matches


if __name__ == "__main__":
    import rasterio

    with rasterio.open("data/raw/test_dummy.tif") as src:
        img1 = src.read(1)

    # Synthetic "pair" banate hain — img1 ko halka rotate/shift karke img2 banao
    # (dual-track validation jaisa concept, filhaal sirf testing ke liye)
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