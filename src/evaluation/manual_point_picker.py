import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime


def pick_points(image, title="Click matching points, then close the window"):
    """
    Image dikhata hai aur user ko points click karne deta hai.
    Har click ek control point maana jata hai.
    
    Args:
        image: numpy array (grayscale image)
        title: window ka title/instruction
    
    Returns:
        List of (x, y) tuples — jo bhi points click kiye gaye
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    
    points = plt.ginput(n=-1, timeout=0, show_clicks=True)
    plt.close(fig)
    
    return points


def save_points(points, person_name, output_dir="outputs/manual_validation"):
    """Points ko JSON file mein save karta hai, person ke naam ke saath."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/{person_name}_points.json"
    
    data = {
        "person": person_name,
        "timestamp": datetime.now().isoformat(),
        "points": points,
    }
    
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ {len(points)} points saved to {filename}")
    return filename


def compare_agreement(file1, file2):
    """
    Do logon ke saved points compare karta hai — mean pixel disagreement nikalta hai.
    Assumption: dono ne same number of points, same order mein click kiya.
    """
    with open(file1) as f:
        data1 = json.load(f)
    with open(file2) as f:
        data2 = json.load(f)
    
    points1 = np.array(data1["points"])
    points2 = np.array(data2["points"])
    
    if len(points1) != len(points2):
        print(f"⚠️ Warning: {data1['person']} ne {len(points1)} points diye, "
              f"{data2['person']} ne {len(points2)} — count match nahi karta!")
        min_len = min(len(points1), len(points2))
        points1 = points1[:min_len]
        points2 = points2[:min_len]
    
    diffs = np.linalg.norm(points1 - points2, axis=1)
    mean_disagreement = np.mean(diffs)
    max_disagreement = np.max(diffs)
    
    print(f"\n📊 Agreement between {data1['person']} and {data2['person']}:")
    print(f"  Points compared: {len(points1)}")
    print(f"  Mean pixel disagreement: {mean_disagreement:.2f} px")
    print(f"  Max pixel disagreement: {max_disagreement:.2f} px")
    
    return {
        "mean_disagreement": float(mean_disagreement),
        "max_disagreement": float(max_disagreement),
        "num_points_compared": int(len(points1)),
    }


if __name__ == "__main__":
    import rasterio

    with rasterio.open("data/raw/test_dummy.tif") as src:
        img = src.read(1)

    person_name = input("Apna naam likho (jaise 'shaket' ya 'gaurav'): ").strip().lower()

    print(f"\n{person_name}, image pe kam se kam 5 identifiable points click karo "
          f"(jaise corners, craters, ya distinct features). Window band karo jab done ho.")

    points = pick_points(img, title=f"{person_name} - Click points, close window when done")
    save_points(points, person_name)