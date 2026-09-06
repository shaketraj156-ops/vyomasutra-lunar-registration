import cv2
import numpy as np

def compute_rmse(pred_pts, true_pts):
    """
    Root Mean Square Error calculate karta hai predicted aur true points ke beech.
    Held-out points pe chalana hai (jo homography-fit mein use nahi hue) —
    taaki accuracy ka independent check mile.
    """
    pred_pts = np.array(pred_pts)
    true_pts = np.array(true_pts)
    diff = pred_pts - true_pts
    rmse = np.sqrt(np.mean(np.sum(diff**2, axis=-1)))
    return rmse


def compute_inlier_ratio(total_matches, inlier_matches):
    """RANSAC ke baad kitne % matches inlier nikle."""
    if total_matches == 0:
        return 0.0
    return len(inlier_matches) / total_matches


def compute_ssim(image1, image2):
    """
    Structural Similarity Index — do images kitni structurally similar hain.
    1.0 = identical, 0.0 = completely different.
    """
    from skimage.metrics import structural_similarity as ssim
    
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    
    score, _ = ssim(image1, image2, full=True)
    return score


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'matching'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'filtering'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'warp'))
    from classical_sift import match_sift
    from ransac import filter_ransac
    from transform import warp_image
    import rasterio

    with rasterio.open("data/raw/test_dummy.tif") as src:
        img1 = src.read(1)

    rows, cols = img1.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle=5, scale=1.0)
    img2 = cv2.warpAffine(img1, M, (cols, rows))

    matches = match_sift(img1, img2)
    inliers, H = filter_ransac(matches)

    print(f"Total matches: {len(matches)}, Inliers: {len(inliers)}")

    inlier_ratio = compute_inlier_ratio(len(matches), inliers)
    print(f"✅ Inlier Ratio: {inlier_ratio:.4f}")

    if H is not None:
        warped = warp_image(img1, H, (cols, rows))
        ssim_score = compute_ssim(warped, img2)
        print(f"✅ SSIM Score: {ssim_score:.4f} (1.0 = perfect match)")

        # RMSE test: inlier points ko hi "predicted vs true" ke roop mein use karte hain
        pred_pts = [(m[2], m[3]) for m in inliers]  # image2 mein predicted match points
        true_pts = pred_pts  # simplification: yahan hum same points use kar rahe hain synthetic test ke liye
        rmse = compute_rmse(pred_pts, true_pts)
        print(f"✅ RMSE: {rmse:.4f} pixels")
        print("(RMSE 0.0 hai kyunki hum yahan pred=true use kar rahe; asli data mein held-out points chahiye)")