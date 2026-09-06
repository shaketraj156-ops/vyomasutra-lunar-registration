import numpy as np

def weighted_distribution_score(matches, image_shape, grid=(8, 8)):
    """
    Match points image mein kitne evenly spread hain, uska score deta hai.
    Ye named differentiator hai — sirf "kitne matches hain" nahi dekhta,
    balki "kitne acche matches poore image mein failey hain" dekhta hai.
    
    200 clustered points (ek hi corner mein) = low score
    80 evenly-spread points = high score
    
    Args:
        matches: List[(x1, y1, x2, y2, confidence)] — SIFT ya LightGlue se
        image_shape: (height, width) tuple
        grid: image ko kitne rows x cols mein baantna hai (default 8x8 = 64 cells)
    
    Returns:
        score: 0.0 se 1.0 ke beech — jitna zyada, utna better spread + confidence
    """
    if not matches:
        return 0.0
    
    h, w = image_shape[:2]
    gh, gw = h / grid[0], w / grid[1]
    
    # Har match kis grid-cell mein padta hai, wo track karo
    cells_hit = set()
    for x1, y1, x2, y2, conf in matches:
        # x1, y1 image1 ke coordinates hain — inhi ko grid mein map karte hain
        cell_row = min(int(y1 // gh), grid[0] - 1)
        cell_col = min(int(x1 // gw), grid[1] - 1)
        cells_hit.add((cell_row, cell_col))
    
    # Coverage: kitne % grid cells mein kam se kam ek match hai
    coverage = len(cells_hit) / (grid[0] * grid[1])
    
    # Average confidence saare matches ka
    mean_conf = sum(conf for *_, conf in matches) / len(matches)
    
    # Dono ko combine karo (weights baad mein tune kar sakte ho)
    score = 0.5 * coverage + 0.5 * mean_conf
    
    return score


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'matching'))
    from classical_sift import match_sift
    import rasterio
    import cv2

    with rasterio.open("data/raw/test_dummy.tif") as src:
        img1 = src.read(1)

    rows, cols = img1.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle=5, scale=1.0)
    img2 = cv2.warpAffine(img1, M, (cols, rows))

    matches = match_sift(img1, img2)
    print(f"Total matches: {len(matches)}")

    score = weighted_distribution_score(matches, img1.shape, grid=(8, 8))
    print(f"\n✅ Weighted Distribution Score: {score:.4f}")
    print("(0.0 = bahut bura spread/confidence, 1.0 = perfect spread + high confidence)")

    # Comparison ke liye: agar matches sirf ek jagah cluster ho jayein toh?
    if matches:
        clustered_matches = [matches[0]] * len(matches)  # sab same point pe cluster kiya
        clustered_score = weighted_distribution_score(clustered_matches, img1.shape, grid=(8, 8))
        print(f"\nAgar sab matches ek hi jagah cluster hote: {clustered_score:.4f}")
        print("(Ye dikhata hai clustering score ko kaise neeche le jaati hai)")