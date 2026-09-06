import cv2
import numpy as np

def resample_image(image, target_size):
    """
    Image ko target size (width, height) pe resize karta hai.
    
    OHRC (0.25m/pixel) aur IIRS (80m/pixel) ke beech 300x ka gap hai,
    isliye ek hi fixed scale pe force-match nahi karte yahan —
    ye function sirf working resolution set karta hai, tiling
    (baad mein Gaurav ke Track B mein) extreme gaps handle karega.
    
    Args:
        image: input image (numpy array)
        target_size: (width, height) tuple
    
    Returns:
        Resized image
    """
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    return resized


if __name__ == "__main__":
    import rasterio
    
    with rasterio.open("data/raw/test_dummy.tif") as src:
        img = src.read(1)
    
    print(f"Original image shape: {img.shape}")
    
    # Test: 200x200 pe resample karo (test ke liye koi bhi target size)
    target_size = (200, 200)
    resized = resample_image(img, target_size)
    
    print(f"Resampled image shape: {resized.shape}")
    print("✅ Resample ho gaya, koi error nahi!")