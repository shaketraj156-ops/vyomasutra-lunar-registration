import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin


def warp_image(image, homography_matrix, output_shape):
    """
    Homography matrix use karke image ko warp/align karta hai.
    Ye rough/preview version hai — sirf numpy array return karta hai.

    Returns:
        warped numpy array, ya None agar warp fail ho jaye (crash nahi karta)
    """
    # ---- NAYA: input validation ----
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        print("⚠️  warp_image: input image invalid/empty hai")
        return None

    if homography_matrix is None:
        print("⚠️  warp_image: homography_matrix None hai")
        return None

    homography_matrix = np.asarray(homography_matrix)
    if homography_matrix.shape != (3, 3):
        print(f"⚠️  warp_image: homography 3x3 hona chahiye, mila shape={homography_matrix.shape}")
        return None

    if len(output_shape) != 2 or output_shape[0] <= 0 or output_shape[1] <= 0:
        print(f"⚠️  warp_image: output_shape invalid hai: {output_shape}")
        return None

    try:
        warped = cv2.warpPerspective(image, homography_matrix, output_shape)
    except cv2.error as e:
        # ---- NAYA: cv2 warp real data pe fail ho sakta hai (bad dtype, singular matrix, etc.) ----
        print(f"⚠️  warp_image: cv2.warpPerspective fail hua: {e}")
        return None

    return warped


def export_geotiff(warped_image, output_path, crs='EPSG:4326',
                    origin_x=0.0, origin_y=0.0, pixel_size=1.0):
    """
    Warped image ko FINAL GeoTIFF format mein save karta hai.

    Returns:
        True agar successful save hui, False agar error
    """
    # ---- NAYA: warped_image None/empty aa sakta hai agar warp_image() upar fail hui ----
    if warped_image is None:
        print("❌ GeoTIFF export failed: warped_image None hai (warp step pehle fail hua hoga)")
        return False

    if not isinstance(warped_image, np.ndarray) or warped_image.size == 0:
        print("❌ GeoTIFF export failed: warped_image empty/invalid hai")
        return False

    # ---- NAYA: pixel_size 0 ya negative hone se transform garbage banta hai ----
    if pixel_size <= 0:
        print(f"❌ GeoTIFF export failed: pixel_size positive hona chahiye, mila {pixel_size}")
        return False

    try:
        height, width = warped_image.shape
        transform = from_origin(origin_x, origin_y, pixel_size, pixel_size)

        if warped_image.dtype != np.uint8:
            warped_image = cv2.normalize(warped_image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=warped_image.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(warped_image, 1)

        return True
    except Exception as e:
        print(f"❌ GeoTIFF export failed: {e}")
        return False


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'matching'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'filtering'))
    from classical_sift import match_sift
    from ransac import filter_ransac

    with rasterio.open("data/raw/test_dummy.tif") as src:
        img1 = src.read(1)

    rows, cols = img1.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle=5, scale=1.0)
    img2 = cv2.warpAffine(img1, M, (cols, rows))

    matches = match_sift(img1, img2)
    inliers, H = filter_ransac(matches)

    if H is not None:
        warped = warp_image(img1, H, (cols, rows))

        if warped is not None:
            os.makedirs("outputs/geotiff", exist_ok=True)
            output_path = "outputs/geotiff/final_registered.tif"

            success = export_geotiff(
                warped, output_path,
                crs='EPSG:4326',
                origin_x=45.0,
                origin_y=-10.0,
                pixel_size=0.001
            )

            if success:
                print(f"✅ Final GeoTIFF saved: {output_path}")
                with rasterio.open(output_path) as check:
                    print(f"\nVerification:")
                    print(f"  CRS: {check.crs}")
                    print(f"  Transform: {check.transform}")
                    print(f"  Size: {check.width} x {check.height}")
            else:
                print("❌ Export fail ho gaya")
        else:
            print("❌ Warp fail ho gaya, export skip kar rahe")
    else:
        print("❌ Homography nahi mila, export skip kar rahe")