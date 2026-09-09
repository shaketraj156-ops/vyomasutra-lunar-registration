import rasterio
import os

# ---- NAYA: valid dtype list, real sensor data ke liye ----
VALID_DTYPES = ('uint8', 'uint16', 'int16', 'int32', 'float32', 'float64')


def validate_image_file(filepath):
    """
    Ek file kholo aur DEEP validation karo — sirf info print nahi,
    actual problems flag karo jo pipeline ko crash kar sakte hain.

    Returns:
        (is_valid: bool, info: dict, issues: List[str])
    """
    issues = []
    info = {}

    if not os.path.exists(filepath):
        return False, {}, [f"File exist hi nahi karti: {filepath}"]

    if os.path.getsize(filepath) == 0:
        return False, {}, [f"File 0 bytes ki hai (empty/corrupt download): {filepath}"]

    try:
        with rasterio.open(filepath) as src:
            info = {
                "crs": src.crs,
                "bounds": src.bounds,
                "dtype": src.dtypes,
                "nodata": src.nodata,
                "width": src.width,
                "height": src.height,
                "bands": src.count,
            }

            # ---- NAYA: CRS missing hai toh georeferencing fail hogi downstream ----
            if src.crs is None:
                issues.append("CRS missing hai — georeferencing/warp step fail ho sakta hai")

            # ---- NAYA: zero-size image (corrupt download common issue) ----
            if src.width == 0 or src.height == 0:
                issues.append(f"Image ka size 0 hai (width={src.width}, height={src.height})")

            # ---- NAYA: no bands ----
            if src.count == 0:
                issues.append("Image mein koi band hi nahi hai")

            # ---- NAYA: unsupported/unexpected dtype ----
            for dt in src.dtypes:
                if dt not in VALID_DTYPES:
                    issues.append(f"Unexpected dtype '{dt}' — SIFT/CLAHE mein normalize karna padega")

            # ---- NAYA: pehla band fully readable hai ya nahi (corrupt data catch) ----
            try:
                band1 = src.read(1)
                if band1.size == 0:
                    issues.append("Band 1 empty array return kar raha hai")
                elif band1.max() == band1.min():
                    issues.append("Band 1 mein saare pixels same value ke hain (blank/corrupt image ho sakti hai)")
            except Exception as e:
                issues.append(f"Band 1 read nahi ho paya: {e}")

    except Exception as e:
        return False, {}, [f"File khulti hi nahi (corrupt/unsupported format): {e}"]

    return (len(issues) == 0), info, issues


def check_file(filepath):
    """Backward-compatible wrapper — purana CLI print behavior, ab validate_image_file use karta hai."""
    is_valid, info, issues = validate_image_file(filepath)

    if info:
        print(f"\n{'✅' if is_valid else '⚠️ '} {os.path.basename(filepath)}")
        print(f"   CRS (coordinate system): {info.get('crs')}")
        print(f"   Bounds (area coverage): {info.get('bounds')}")
        print(f"   Data type: {info.get('dtype')}")
        print(f"   Nodata value: {info.get('nodata')}")
        print(f"   Size: {info.get('width')} x {info.get('height')} pixels")
        print(f"   Bands: {info.get('bands')}")
    else:
        print(f"\n❌ CORRUPT FILE: {os.path.basename(filepath)}")

    for issue in issues:
        print(f"   ⚠️  ISSUE: {issue}")

    # NOTE: pehle sirf "file khuli ya nahi" pe True/False tha.
    # Ab CRS-missing, zero-size, jaise issues bhi False bana dete hain
    # taaki pipeline ko pata chale ki file "technically open" hai par "usable" nahi.
    return is_valid


def check_folder(folder_path):
    """Poore folder mein saari image files check karo."""
    if not os.path.exists(folder_path):
        print(f"⚠️  Folder nahi mila: {folder_path}")
        return

    valid_extensions = ('.tif', '.tiff', '.img')
    files_found = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]

    if not files_found:
        print(f"⚠️  Koi image file nahi mili {folder_path} mein")
        print("   (.tif, .tiff, .img extensions dhundi ja rahi thi)")
        return

    print(f"📁 {len(files_found)} file(s) mili {folder_path} mein\n")
    good_count = 0
    for file in files_found:
        if check_file(os.path.join(folder_path, file)):
            good_count += 1

    print(f"\n{'='*40}")
    print(f"Summary: {good_count}/{len(files_found)} files sahi hain")
    print(f"{'='*40}")


if __name__ == "__main__":
    print("🔍 Checking data/raw folder...")
    check_folder("data/raw")

    print("\n🔍 Checking data/reference folder...")
    check_folder("data/reference")