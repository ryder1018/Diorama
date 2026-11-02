"""Step 1: see the photo. Muse Spark chooses what to look for, SAM 3.1 segments it.

Writes, for a photo named <base>:

  <out_dir>/<base>_depth_mask.npy        SAM 3.1 label map (0 = no object)
  <out_dir>/<base>_sam31_objects.json    named objects, concepts, timings

The photo itself is resized in place to `output_width`, the size every later
step (layers, Unity, showcase) works at; SAM 3.1 sees a smaller copy.
"""

import json
import os
import time

import cv2
import numpy as np

from sam31_segmentation import generate_sam31_label_map

OUTPUT_WIDTH = int(os.getenv("PIPELINE_OUTPUT_WIDTH", "1024"))
INFERENCE_WIDTH = int(os.getenv("PIPELINE_INFERENCE_WIDTH", "768"))


def fit_width(image: np.ndarray, width: int) -> np.ndarray:
    height, current = image.shape[:2]
    if current == width:
        return image
    return cv2.resize(image, (width, max(1, int(height * width / current))), interpolation=cv2.INTER_AREA)


def output_paths(image_path: str, out_dir: str) -> dict[str, str]:
    base = os.path.splitext(os.path.basename(image_path))[0]
    return {
        "labels": os.path.join(out_dir, f"{base}_depth_mask.npy"),
        "objects": os.path.join(out_dir, f"{base}_sam31_objects.json"),
        "depth": os.path.join(out_dir, f"{base}_depth_16bit.png"),  # written by depth_order.py
    }


def analyze_photo(
    image_path: str,
    out_dir: str,
    output_width: int = OUTPUT_WIDTH,
    inference_width: int = INFERENCE_WIDTH,
) -> dict[str, str]:
    """Segment one photo into named SAM 3.1 regions; returns the paths of the files written."""
    photo = cv2.imread(image_path)
    if photo is None:
        raise ValueError(f"Cannot read image: {image_path}")
    os.makedirs(out_dir, exist_ok=True)
    paths = output_paths(image_path, out_dir)

    resized = fit_width(photo, output_width)
    if resized.shape != photo.shape:
        cv2.imwrite(image_path, resized)
        print(f"  ✔ Photo resized to {resized.shape[1]}x{resized.shape[0]}")
    height, width = resized.shape[:2]
    small = fit_width(photo, inference_width)

    labels, count, objects, info = generate_sam31_label_map(small)
    labels = cv2.resize(labels.astype(np.uint16), (width, height), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    np.save(paths["labels"], labels)

    small_h, small_w = small.shape[:2]
    for obj in objects:
        x1, y1, x2, y2 = obj.pop("box")
        obj["box"] = [x1 / small_w, y1 / small_h, x2 / small_w, y2 / small_h]
        obj["area_ratio"] = obj.pop("area") / float(small_w * small_h)
    with open(paths["objects"], "w", encoding="utf-8") as f:
        json.dump(
            {"width": width, "height": height, "concepts": info["concepts"], "timings": info["timings"], "objects": objects},
            f,
            indent=2,
        )
    print(f"  ✔ SAM 3.1 found {count} region(s)")
    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Segment one photo with Muse Spark concepts + SAM 3.1.")
    parser.add_argument("image_path")
    parser.add_argument("--out_dir", default="./outputs/inference_results")
    args = parser.parse_args()
    started = time.perf_counter()
    analyze_photo(args.image_path, args.out_dir)
    print(f"Done in {time.perf_counter() - started:.1f}s")
