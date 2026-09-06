import rasterio
import os


def check_file(filepath):
    """Ek file kholo aur uski basic details print karo. Agar file corrupt hai, ye pakad legi."""
    try:
        with rasterio.open(filepath) as src:
            print(f"\n✅ {os.path.basename(filepath)}")
            print(f"   CRS (coordinate system): {src.crs}")
            print(f"   Bounds (area coverage): {src.bounds}")
            print(f"   Data type: {src.dtypes}")
            print(f"   Nodata value: {src.nodata}")
            print(f"   Size: {src.width} x {src.height} pixels")
            print(f"   Bands: {src.count}")
            return True
    except Exception as e:
        print(f"\n❌ CORRUPT FILE: {os.path.basename(filepath)}")
        print(f"   Error: {e}")
        return False


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