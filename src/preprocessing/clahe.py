import cv2
import numpy as np

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) apply karta hai.
    Ye lunar images mein extreme lighting/shadow difference ko fix karta hai.
    
    Args:
        image: input image (grayscale, numpy array)
        clip_limit: contrast limit (sensors.yaml se aayega, sensor ke hisaab se)
        tile_grid_size: kitne chhote blocks mein image ko baant kar process kare
    
    Returns:
        CLAHE-enhanced image
    """
    # Agar image already 8-bit nahi hai, usse normalize karo
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image)
    return enhanced


if __name__ == "__main__":
    # Test: apni test_dummy.tif pe try karo
    import rasterio
    
    with rasterio.open("data/raw/test_dummy.tif") as src:
        img = src.read(1)  # pehla band padho
    
    print(f"Original image shape: {img.shape}, dtype: {img.dtype}")
    
    enhanced = apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8))
    
    print(f"Enhanced image shape: {enhanced.shape}, dtype: {enhanced.dtype}")
    print("✅ CLAHE apply ho gaya, koi error nahi!")