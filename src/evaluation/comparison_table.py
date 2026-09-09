import json
import os
from datetime import datetime


def generate_comparison_table(our_results, output_path="outputs/reports/comparison_table.json"):
    """
    Hamare results ko Makharia et al. (arXiv:2509.04775) baseline paper ke
    reported numbers ke saath side-by-side rakhta hai.
    """
    baseline_data = {
        "OHRC-NAC Equatorial": {
            "SIFT": {"rmse_x": 3.6096, "rmse_y": 5.9558, "time_s": 678.20},
            "AKAZE": {"rmse_x": 3.1189, "rmse_y": 4.7096, "time_s": 737.17},
            "SuperGlue": {"rmse_x": 0.6249, "rmse_y": 0.5718, "time_s": 3.809},
        },
        "IIRS-WAC Equatorial": {
            "SIFT": {"rmse_x": 0.6879, "rmse_y": 1.1066, "time_s": 0.1207},
            "SuperGlue": {"rmse_x": 0.5069, "rmse_y": 0.6167, "time_s": 0.818},
        },
        "IIRS-WAC Polar": {
            "SIFT": {"rmse_x": 2.0085, "rmse_y": 0.4050, "time_s": 0.1607},
            "SuperGlue": {"rmse_x": 0.7681, "rmse_y": 0.9267, "time_s": 0.774},
        },
    }

    comparison = {
        "generated_at": datetime.now().isoformat(),
        "baseline_paper": "Makharia et al., arXiv:2509.04775",
        "baseline_numbers": baseline_data,
        "our_results": our_results,
        "notes": [
            "Our SIFT and LightGlue numbers are from a synthetic (self-rotated) test image, "
            "not the paper's real lunar datasets — direct RMSE comparison is not yet apples-to-apples.",
            "LightGlue is the successor architecture to SuperGlue (used in the paper) — "
            "expected to perform similarly or better once tested on matching real sensor data.",
            "Full apples-to-apples comparison requires running on the same scene IDs used in the "
            "baseline paper, once real OHRC/TMC-2/IIRS data is available.",
        ],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)

    return comparison


if __name__ == "__main__":
    our_results = {
        "test_type": "synthetic (5-degree rotation, test_dummy.tif)",
        "sift_matches": 22,
        "lightglue_matches": 738,
        "combined_matches": 760,
        "inliers_after_ransac": 754,
        "inlier_ratio": 0.9921,
        "weighted_distribution_score": 0.9116,
        "ssim_score": 0.9909,
    }

    comparison = generate_comparison_table(our_results)

    print("✅ Comparison table generated:\n")
    print(json.dumps(comparison, indent=2))