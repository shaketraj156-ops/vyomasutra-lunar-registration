import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin


def warp_image(image, homography_matrix, output_shape):
    """
    Homography matrix use karke image ko warp/align karta hai.
    Ye rough/preview version hai — sirf numpy array return karta hai.
    """
    warped = cv2.warpPerspective(image, homography_matrix, output_shape)
    return warped


def export_geotiff(warped_image, output_path, crs='EPSG:4326',
                    origin_x=0.0, origin_y=0.0, pixel_size=1.0):
    """
    Warped image ko FINAL GeoTIFF format mein save karta hai —
    CRS aur affine transform preserve karke. Ye PNG nahi hai,
    real GIS software (QGIS, etc.) mein seedha khulega.

    Args:
        warped_image: warp_image() se aaya numpy array
        output_path: kaha save karni hai (e.g. "outputs/geotiff/result.tif")
        crs: coordinate reference system
        origin_x, origin_y: reference image ka top-left real-world coordinate
        pixel_size: reference image ka pixel resolution (meters/pixel)

    Returns:
        True agar successful save hui, False agar error
    """
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

            # Verify: file ko wapas khol ke confirm karo CRS/transform sahi hai
            with rasterio.open(output_path) as check:
                print(f"\nVerification:")
                print(f"  CRS: {check.crs}")
                print(f"  Transform: {check.transform}")
                print(f"  Size: {check.width} x {check.height}")
        else:
            print("❌ Export fail ho gaya")
    else:
        print("❌ Homography nahi mila, export skip kar rahe")