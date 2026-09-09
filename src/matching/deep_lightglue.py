
from typing import List, Tuple
import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

Match = Tuple[float, float, float, float, float]


def match_images(source_path: str, reference_path: str) -> List[Match]:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image0 = load_image(source_path).to(device)
    image1 = load_image(reference_path).to(device)

    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    with torch.inference_mode():
        feats0 = extractor.extract(image0)
        feats1 = extractor.extract(image1)

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

        result.append(
            (float(x1), float(y1), float(x2), float(y2), confidence)
        )

    return result
