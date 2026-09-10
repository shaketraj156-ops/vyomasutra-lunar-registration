"""
deep_lightglue.py
Deep learning feature matcher using SuperPoint and LightGlue.
Supports both in-memory numpy arrays and disk file paths.
Caches neural network models to eliminate re-instantiation overhead.
"""

from typing import List, Tuple, Union
import numpy as np
import torch

try:
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
    LIGHTGLUE_AVAILABLE = True
except ImportError:
    LIGHTGLUE_AVAILABLE = False

Match = Tuple[float, float, float, float, float]

# Global cache for models to avoid reloading weights on every match
_EXTRACTOR = None
_MATCHER = None
_DEVICE = None


def get_models(max_keypoints: int = 2048):
    """Lazy initialization and caching of SuperPoint and LightGlue models."""
    global _EXTRACTOR, _MATCHER, _DEVICE
    if not LIGHTGLUE_AVAILABLE:
        raise RuntimeError("LightGlue package is not installed. Install via git+https://github.com/cvg/LightGlue.git")

    if _DEVICE is None:
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if _EXTRACTOR is None or _MATCHER is None:
        _EXTRACTOR = SuperPoint(max_num_keypoints=max_keypoints).eval().to(_DEVICE)
        _MATCHER = LightGlue(features="superpoint").eval().to(_DEVICE)

    return _EXTRACTOR, _MATCHER, _DEVICE


def _numpy_to_tensor(image: np.ndarray, device: str) -> torch.Tensor:
    """Converts a 2D grayscale numpy array to a normalized (1, 1, H, W) PyTorch float tensor."""
    if image.ndim == 3:
        import cv2
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.dtype != np.float32:
        img_float = image.astype(np.float32)
        img_min = float(np.nanmin(img_float))
        img_max = float(np.nanmax(img_float))
        if img_max > img_min:
            img_float = (img_float - img_min) / (img_max - img_min)
        else:
            img_float = np.zeros_like(img_float)
    else:
        img_float = image.copy()
        if img_float.max() > 1.0:
            img_float = img_float / 255.0

    tensor = torch.from_numpy(img_float).unsqueeze(0).unsqueeze(0).to(device)
    return tensor


def match_arrays(image0_np: np.ndarray, image1_np: np.ndarray, max_keypoints: int = 2048) -> List[Match]:
    """
    Matches features between two in-memory numpy grayscale images using SuperPoint + LightGlue.
    
    Args:
        image0_np: Source image (2D numpy array, uint8 or float32)
        image1_np: Reference image (2D numpy array, uint8 or float32)
        max_keypoints: Max keypoints to extract
        
    Returns:
        List[(x1, y1, x2, y2, confidence)]
    """
    if not LIGHTGLUE_AVAILABLE:
        print("⚠️ LightGlue is not available, returning empty matches.")
        return []

    try:
        extractor, matcher, device = get_models(max_keypoints=max_keypoints)
    except Exception as e:
        print(f"⚠️ Failed to load LightGlue models: {e}")
        return []

    try:
        t0 = _numpy_to_tensor(image0_np, device)
        t1 = _numpy_to_tensor(image1_np, device)

        with torch.inference_mode():
            feats0 = extractor.extract(t0)
            feats1 = extractor.extract(t1)

            matches01 = matcher({
                "image0": feats0,
                "image1": feats1,
            })

        feats0, feats1, matches01 = [
            rbd(x) for x in (feats0, feats1, matches01)
        ]

        keypoints0 = feats0["keypoints"].cpu()
        keypoints1 = feats1["keypoints"].cpu()
        matches = matches01["matches"].cpu()
        scores = matches01["scores"].cpu()

        result = []
        for i, (idx0, idx1) in enumerate(matches):
            x1, y1 = keypoints0[idx0].tolist()
            x2, y2 = keypoints1[idx1].tolist()
            confidence = float(scores[i])
            result.append((float(x1), float(y1), float(x2), float(y2), confidence))

        return result
    except Exception as e:
        print(f"⚠️ LightGlue matching failed: {e}")
        return []


def match_images(source_path: str, reference_path: str, max_keypoints: int = 2048) -> List[Match]:
    """Backward-compatible wrapper that loads images from disk using rasterio or cv2, then matches."""
    import rasterio

    try:
        with rasterio.open(source_path) as src:
            img0 = src.read(1)
        with rasterio.open(reference_path) as src:
            img1 = src.read(1)
    except Exception:
        import cv2
        img0 = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)

    return match_arrays(img0, img1, max_keypoints=max_keypoints)
