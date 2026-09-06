import rasterio
from rasterio.transform import from_origin

def set_georeference(image, output_path, crs='EPSG:4326', 
                      origin_x=0.0, origin_y=0.0, pixel_size=1.0):
    """
    Image ko georeference karke ek nayi GeoTIFF file mein save karta hai.
    
    Args:
        image: numpy array (processed image, jaise CLAHE/resample ke baad)
        output_path: kahan save karni hai (e.g. "data/processed/output.tif")
        crs: coordinate reference system (default: WGS84, lat/lon)
        origin_x, origin_y: top-left corner ka real-world coordinate
        pixel_size: har pixel kitni real-world distance cover karta hai
    
    Returns:
        True agar successful, False agar error
    """
    try:
        height, width = image.shape
        transform = from_origin(origin_x, origin_y, pixel_size, pixel_size)
        
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=image.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(image, 1)
        
        return True
    except Exception as e:
        print(f"❌ Georeferencing failed: {e}")
        return False


def get_georef_info(filepath):
    """Kisi existing file ka CRS aur transform info print karta hai."""
    with rasterio.open(filepath) as src:
        print(f"CRS: {src.crs}")
        print(f"Transform: {src.transform}")
        print(f"Bounds: {src.bounds}")


if __name__ == "__main__":
    import numpy as np
    import os
    
    # Test: ek dummy processed image banao aur georeference karo
    test_image = np.random.randint(0, 255, (100, 100), dtype='uint8')
    
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/test_georef.tif"
    
    success = set_georeference(
        test_image, 
        output_path,
        crs='EPSG:4326',
        origin_x=45.0,   # koi bhi test coordinate
        origin_y=-10.0,
        pixel_size=0.001
    )
    
    if success:
        print(f"✅ Georeferenced file bani: {output_path}\n")
        get_georef_info(output_path)
    else:
        print("❌ Kuch galat hua")