"""
create_sample_data.py
Generates realistic lunar surface test pairs with craters, terrain noise,
sun-angle shading variations, and known ground-truth affine/projective transformations.
Ensures the app has built-in instant demo datasets.
"""

import os
import cv2
import numpy as np
import rasterio
import rasterio.crs
from rasterio.transform import from_origin


def generate_lunar_terrain(width=512, height=512, seed=42):
    """Synthesizes a realistic lunar surface with multiple craters and regolith roughness."""
    np.random.seed(seed)

    # Base regolith texture via multi-scale noise
    base = np.zeros((height, width), dtype=np.float32)
    for scale, weight in [(64, 0.4), (32, 0.3), (16, 0.2), (8, 0.1)]:
        noise = cv2.resize(np.random.normal(128, 25, (height // scale, width // scale)).astype(np.float32), (width, height))
        base += noise * weight

    # Add craters with illuminated rims and dark interior shadows
    craters = [
        (160, 140, 55, 0.8),
        (340, 190, 70, 0.9),
        (220, 360, 90, 1.0),
        (390, 380, 45, 0.7),
        (100, 320, 35, 0.6),
        (270, 240, 25, 0.5),
        (430, 110, 30, 0.6),
        (70, 80, 20, 0.5),
    ]

    for cx, cy, radius, depth in craters:
        y, x = np.ogrid[:height, :width]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        crater_mask = dist <= radius

        # Shadow direction (lighting from top-left, shadow towards bottom-right)
        dx = (x - cx) / (radius + 1e-5)
        dy = (y - cy) / (radius + 1e-5)
        lighting = 0.5 * dx + 0.5 * dy

        # Depression + Rim
        rim_mask = (dist > radius * 0.85) & (dist <= radius * 1.15)
        base[crater_mask] -= depth * 60.0 * (1.0 - (dist[crater_mask] / radius))
        base[crater_mask] += lighting[crater_mask] * 35.0
        base[rim_mask] += depth * 25.0

    terrain = np.clip(base, 10, 245).astype(np.uint8)
    return terrain


def create_demo_pair(output_dir="data/samples"):
    """Creates a pair of Lunar images (Source & Reference) with known transformation and lighting differences."""
    os.makedirs(output_dir, exist_ok=True)

    # Reference Image
    ref_img = generate_lunar_terrain(512, 512, seed=101)

    # Source Image: Slightly rotated (5.0 deg), translated (8px, -6px), and illumination adjusted
    M = cv2.getRotationMatrix2D((256, 256), angle=5.0, scale=1.0)
    M[0, 2] += 8.0
    M[1, 2] -= 6.0

    src_img = cv2.warpAffine(ref_img, M, (512, 512), borderMode=cv2.BORDER_REFLECT)
    src_img = cv2.convertScaleAbs(src_img, alpha=1.1, beta=-10)

    ref_path = os.path.join(output_dir, "lunar_reference_crater.tif")
    src_path = os.path.join(output_dir, "lunar_source_crater.tif")

    # Lunar IAU2000 Sphere CRS (Radius = 1737.4 km)
    crs = rasterio.crs.CRS.from_proj4("+proj=longlat +R=1737400 +no_defs")
    transform = from_origin(15.2, -8.4, 0.0005, 0.0005)

    for path, img in [(ref_path, ref_img), (src_path, src_img)]:
        with rasterio.open(
            path, 'w',
            driver='GTiff',
            height=512,
            width=512,
            count=1,
            dtype='uint8',
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(img, 1)

    print(f"✅ Created demo sample pair:\n  Reference: {ref_path}\n  Source: {src_path}")
    return src_path, ref_path


if __name__ == "__main__":
    create_demo_pair()
