"""
transform.py
Image warping, checkerboard overlay generation, and GeoTIFF export
with spatial metadata inheritance from reference imagery.
"""

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin
from typing import Optional, Tuple


def warp_image(image: np.ndarray, homography_matrix: np.ndarray, output_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Warps source image onto reference coordinate frame using estimated homography.
    
    Args:
        image: Source grayscale image (2D numpy array)
        homography_matrix: 3x3 homography matrix H
        output_shape: (width, height) of destination canvas (must match reference dimensions!)
        
    Returns:
        Warped numpy array or None if invalid.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        print("[WARNING] warp_image: input image invalid/empty")
        return None

    if homography_matrix is None:
        print("[WARNING] warp_image: homography_matrix is None")
        return None

    homography_matrix = np.asarray(homography_matrix, dtype=np.float64)
    if homography_matrix.shape != (3, 3):
        print(f"[WARNING] warp_image: homography must be 3x3, got {homography_matrix.shape}")
        return None

    # Check for degeneracy via pure 3x3 arithmetic determinant (avoids broken BLAS/LAPACK DLL crashes)
    H = homography_matrix
    det = (H[0, 0] * (H[1, 1] * H[2, 2] - H[1, 2] * H[2, 1])
           - H[0, 1] * (H[1, 0] * H[2, 2] - H[1, 2] * H[2, 0])
           + H[0, 2] * (H[1, 0] * H[2, 1] - H[1, 1] * H[2, 0]))

    if abs(det) < 1e-8:
        print(f"[WARNING] warp_image: degenerate homography matrix (det={det})")
        return None

    if len(output_shape) != 2 or output_shape[0] <= 0 or output_shape[1] <= 0:
        print(f"[WARNING] warp_image: invalid output_shape {output_shape}")
        return None

    try:
        # OpenCV warpPerspective takes dsize as (width, height) = (cols, rows)
        img_contig = np.ascontiguousarray(image)
        warped = cv2.warpPerspective(img_contig, homography_matrix, (int(output_shape[0]), int(output_shape[1])))
        return warped
    except cv2.error as e:
        print(f"[WARNING] warp_image: cv2.warpPerspective failed: {e}")
        return None


def create_checkerboard_overlay(
    image1: np.ndarray,
    image2: np.ndarray,
    tile_size: int = 64
) -> np.ndarray:
    """
    Creates an alternating checkerboard pattern of two registered images.
    If registration is accurate, continuous linear features (like crater rims)
    seamlessly cross tile boundaries.
    
    Args:
        image1: First registered image (e.g. Warped Source)
        image2: Second registered image (Reference Image)
        tile_size: Square tile dimension in pixels
        
    Returns:
        Composite checkerboard image.
    """
    # Match shapes if necessary
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    h, w = min(h1, h2), min(w1, w2)

    img1 = image1[:h, :w]
    img2 = image2[:h, :w]

    # Normalize both to 8-bit for clean visual blend
    if img1.dtype != np.uint8:
        img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Generate checkerboard mask
    grid_y, grid_x = np.indices((h, w))
    tile_idx_y = grid_y // tile_size
    tile_idx_x = grid_x // tile_size
    mask = (tile_idx_y + tile_idx_x) % 2 == 0

    composite = np.where(mask, img1, img2)
    return composite


def export_geotiff(
    warped_image: np.ndarray,
    output_path: str,
    ref_crs = None,
    ref_transform = None,
    default_crs: str = 'IAU2000:30100',
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    pixel_size: float = 1.0
) -> bool:
    """
    Exports warped image to a true GIS GeoTIFF.
    Inherits CRS and Affine Transform directly from reference image whenever available.
    """
    if warped_image is None or not isinstance(warped_image, np.ndarray) or warped_image.size == 0:
        print("[ERROR] GeoTIFF export failed: warped_image empty/invalid")
        return False

    try:
        height, width = warped_image.shape[:2]

        if ref_transform is not None:
            transform = ref_transform
        else:
            transform = from_origin(origin_x, origin_y, pixel_size, pixel_size)

        crs = ref_crs if ref_crs is not None else default_crs

        export_data = warped_image
        if export_data.dtype not in (np.uint8, np.uint16, np.float32):
            export_data = cv2.normalize(export_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=export_data.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(export_data, 1)

        return True
    except Exception as e:
        print(f"[ERROR] GeoTIFF export failed: {e}")
        return False