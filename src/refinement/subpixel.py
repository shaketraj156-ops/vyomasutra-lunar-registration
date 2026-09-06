import cv2
import numpy as np

def refine_subpixel(image, matches, win_size=(5, 5), zero_zone=(-1, -1), max_iter=30, eps=0.001):
    """
    Har matched keypoint ko sub-pixel precision tak refine karta hai.
    Normal matching pixel-level accurate hota hai (jaise pixel 45, 67),
    ye function usse aur zoom karke fraction tak leke jata hai (jaise 45.3, 67.8).
    
    Args:
        image: source image (grayscale, numpy array) jisme points refine karne hain
        matches: List[(x1, y1, x2, y2, confidence)] — original matches
        win_size: search window size around each point
        zero_zone: dead zone size (-1,-1 matlab koi dead zone nahi)
        max_iter, eps: refinement kab stop kare (iterations ya precision)
    
    Returns:
        List[(x1_refined, y1_refined, x2, y2, confidence)] — 
        NOTE: sirf image1 ke points (x1,y1) refine hote hain, kyunki
        refine_subpixel ek hi image ke against chalta hai. Agar dono
        images refine karni hain, function ko dono baar call karo.
    """
    if not matches:
        return []
    
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    
    # cornerSubPix ko float32 points chahiye, shape (N, 1, 2)
    points = np.array([[m[0], m[1]] for m in matches], dtype=np.float32).reshape(-1, 1, 2)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
    refined_points = cv2.cornerSubPix(image, points, win_size, zero_zone, criteria)
    
    refined_matches = []
    for i, (x1, y1, x2, y2, conf) in enumerate(matches):
        new_x1, new_y1 = refined_points[i][0]
        refined_matches.append((float(new_x1), float(new_y1), x2, y2, conf))
    
    return refined_matches


if __name__ == "__main__":
    import numpy as np
    import cv2

    # Ek checkerboard-jaisa pattern banao — real corners ke saath
    img1 = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img1, (20, 20), (80, 80), 255, -1)  # ek safed square
    cv2.rectangle(img1, (40, 40), (60, 60), 0, -1)    # beech mein kaala square

    # Fake matches banao jo square ke corners ke paas hain (thoda off-center)
    fake_matches = [
        (19.7, 19.6, 0, 0, 0.9),   # top-left corner ke paas
        (80.3, 20.4, 0, 0, 0.9),   # top-right corner ke paas
        (40.4, 40.3, 0, 0, 0.9),   # inner corner ke paas
    ]

    print("Testing with a real corner pattern (not random noise):")
    for x, y, *_ in fake_matches:
        print(f"  Original: ({x:.4f}, {y:.4f})")

    refined = refine_subpixel(img1, fake_matches)

    print("\nRefined points:")
    for r in refined:
        print(f"  Refined: ({r[0]:.4f}, {r[1]:.4f})")