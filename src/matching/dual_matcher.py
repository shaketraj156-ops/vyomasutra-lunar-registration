"""
dual_matcher.py
Combines classical (SIFT) and deep learning (LightGlue) matching results
into a single unified list, using the shared schema:
    List[(x1, y1, x2, y2, confidence)]
"""

import os
import rasterio
from src.matching.classical_sift import match_sift
from src.matching.deep_lightglue import match_images as lightglue_match_images


def load_image_array(path):
    """Rasterio se image ko numpy array mein load karo (SIFT ke liye chahiye)."""
    # ---- NAYA: file exist check, warna rasterio ka error message confusing hota hai ----
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file nahi mili: {path}")

    try:
        with rasterio.open(path) as src:
            band1 = src.read(1)
    except Exception as e:
        # ---- NAYA: corrupt/unsupported file ko clear message ke saath re-raise karo ----
        raise RuntimeError(f"'{path}' khulti nahi (corrupt ya unsupported format): {e}")

    if band1.size == 0:
        raise RuntimeError(f"'{path}' mein band 1 empty hai")

    return band1


def get_sift_matches(source_path, reference_path):
    """
    SIFT function image ARRAYS leta hai, paths nahi — isliye pehle
    rasterio se images load karte hain, fir match_sift() ko dete hain.
    """
    img1 = load_image_array(source_path)
    img2 = load_image_array(reference_path)
    matches = match_sift(img1, img2)
    tagged = [(*m, "sift") for m in matches]
    return tagged


def get_lightglue_matches(source_path, reference_path):
    """LightGlue seedha file paths leta hai."""
    # ---- NAYA: file existence yahan bhi check karo, LightGlue ka error kabhi unclear hota hai ----
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source image nahi mili: {source_path}")
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference image nahi mili: {reference_path}")

    matches = lightglue_match_images(source_path, reference_path)
    tagged = [(*m, "lightglue") for m in matches]
    return tagged


def combine_matches(source_path, reference_path, run_sift=True, run_lightglue=True):
    """
    Returns:
        List of tagged matches.
    Raises:
        RuntimeError — agar SIFT aur LightGlue dono fail ho gaye (pehle silently [] return hota tha,
        jo downstream RANSAC ko confusing "0 matches" error deta tha without wajah bataye).
    """
    combined = []
    sift_error = None
    lightglue_error = None

    if run_sift:
        try:
            sift_matches = get_sift_matches(source_path, reference_path)
            print(f"SIFT: {len(sift_matches)} matches found")
            combined.extend(sift_matches)
        except Exception as e:
            sift_error = e
            print(f"⚠️ SIFT matching failed: {e}")

    if run_lightglue:
        try:
            lg_matches = get_lightglue_matches(source_path, reference_path)
            print(f"LightGlue: {len(lg_matches)} matches found")
            combined.extend(lg_matches)
        except Exception as e:
            lightglue_error = e
            print(f"⚠️ LightGlue matching failed: {e}")

    # ---- NAYA: agar dono matchers fail hue (ya dono skip the), clearly batao kyun ----
    if run_sift and run_lightglue and sift_error is not None and lightglue_error is not None:
        raise RuntimeError(
            f"Dono matchers fail ho gaye — SIFT: {sift_error} | LightGlue: {lightglue_error}"
        )

    print(f"Total combined matches: {len(combined)}")
    return combined


def strip_source_tag(tagged_matches):
    """Downstream stages (RANSAC, distribution_score) ke liye 'source' tag hatao."""
    return [m[:5] for m in tagged_matches]


if __name__ == "__main__":
    source_path = "data/raw/test_dummy.tif"
    reference_path = "data/raw/test_dummy.tif"  # abhi ke liye same file, test ke liye

    combined = combine_matches(source_path, reference_path)

    print("\nFirst 5 combined matches (with source tag):")
    for m in combined[:5]:
        print(m)

    clean_matches = strip_source_tag(combined)
    print(f"\nClean matches ready for downstream pipeline: {len(clean_matches)}")