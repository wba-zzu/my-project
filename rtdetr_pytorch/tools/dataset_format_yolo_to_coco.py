#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert VisDrone2019 YOLO-format labels to COCO JSON.

YOLO label format:
    class_id x_center y_center width height

Example:
    3 0.376563 0.419444 0.042708 0.031481

Dataset structure:

datasets/
└── visdrone2019/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── annotations/
        ├── instances_train.json
        └── instances_val.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


# ============================================================
# VisDrone 10 classes
#
# IMPORTANT:
# RT-DETR uses category_id directly when:
#
#     remap_mscoco_category: False
#
# Therefore keep category IDs as 0~9, exactly matching YOLO.
# ============================================================

CLASS_NAMES = [
    "pedestrian",       # 0
    "people",           # 1
    "bicycle",          # 2
    "car",              # 3
    "van",              # 4
    "truck",            # 5
    "tricycle",         # 6
    "awning-tricycle",  # 7
    "bus",              # 8
    "motor",            # 9
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def get_image_files(image_dir: Path):
    """Recursively find all supported image files."""

    files = []

    for path in image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(path)

    return sorted(files)


def yolo_box_to_coco(
    x_center,
    y_center,
    box_width,
    box_height,
    image_width,
    image_height,
):
    """
    Convert normalized YOLO bbox:

        cx, cy, w, h

    to COCO absolute bbox:

        x_min, y_min, width, height

    Bounding boxes are clipped to image boundaries.
    """

    # YOLO normalized -> absolute coordinates
    cx = x_center * image_width
    cy = y_center * image_height

    bw = box_width * image_width
    bh = box_height * image_height

    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0

    # Clip to image boundaries
    x1 = max(0.0, min(x1, float(image_width)))
    y1 = max(0.0, min(y1, float(image_height)))
    x2 = max(0.0, min(x2, float(image_width)))
    y2 = max(0.0, min(y2, float(image_height)))

    bw = x2 - x1
    bh = y2 - y1

    if bw <= 0 or bh <= 0:
        return None

    return [x1, y1, bw, bh]


def convert_split(base_dir: Path, split: str):
    """Convert one split: train / val / test."""

    image_dir = base_dir / "images" / split
    label_dir = base_dir / "labels" / split

    annotation_dir = base_dir / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)

    output_json = annotation_dir / f"instances_{split}.json"

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory does not exist:\n{image_dir}"
        )

    if not label_dir.exists():
        raise FileNotFoundError(
            f"Label directory does not exist:\n{label_dir}"
        )

    image_files = get_image_files(image_dir)

    if len(image_files) == 0:
        raise RuntimeError(
            f"No images found in:\n{image_dir}"
        )

    print("=" * 80)
    print(f"Converting split: {split}")
    print(f"Images : {image_dir}")
    print(f"Labels : {label_dir}")
    print(f"Output : {output_json}")
    print(f"Found images: {len(image_files)}")
    print("=" * 80)

    coco = {
        "info": {
            "description": "VisDrone2019 converted from YOLO to COCO",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": class_id,
                "name": class_name,
                "supercategory": "object",
            }
            for class_id, class_name in enumerate(CLASS_NAMES)
        ],
    }

    annotation_id = 1

    missing_labels = []
    empty_labels = []
    invalid_lines = []

    class_counter = Counter()

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for image_id, image_path in enumerate(image_files, start=1):

        relative_image_path = image_path.relative_to(image_dir)

        # Corresponding label path
        label_path = (
            label_dir
            / relative_image_path
        ).with_suffix(".txt")

        # Read image width / height
        try:
            with Image.open(image_path) as img:
                image_width, image_height = img.size

        except Exception as exc:
            print(
                f"[ERROR] Cannot read image: {image_path}\n"
                f"        {exc}"
            )
            continue

        # Add image info to COCO
        coco["images"].append(
            {
                "id": image_id,
                "file_name": relative_image_path.as_posix(),
                "width": image_width,
                "height": image_height,
            }
        )

        # ----------------------------------------------------
        # No label file
        #
        # We still keep the image in COCO.
        # It will be treated as an image with zero objects.
        # ----------------------------------------------------

        if not label_path.exists():
            missing_labels.append(
                relative_image_path.as_posix()
            )
            continue

        lines = label_path.read_text(
            encoding="utf-8"
        ).strip().splitlines()

        if len(lines) == 0:
            empty_labels.append(
                relative_image_path.as_posix()
            )
            continue

        # ----------------------------------------------------
        # Parse YOLO labels
        # ----------------------------------------------------

        for line_number, line in enumerate(lines, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            # Standard YOLO detection format must contain 5 values
            if len(parts) != 5:
                invalid_lines.append(
                    {
                        "label": str(label_path),
                        "line": line_number,
                        "reason": f"Expected 5 values, got {len(parts)}",
                        "content": line,
                    }
                )
                continue

            try:
                class_id = int(float(parts[0]))

                x_center = float(parts[1])
                y_center = float(parts[2])
                box_width = float(parts[3])
                box_height = float(parts[4])

            except ValueError:
                invalid_lines.append(
                    {
                        "label": str(label_path),
                        "line": line_number,
                        "reason": "Cannot parse numeric values",
                        "content": line,
                    }
                )
                continue

            # ------------------------------------------------
            # Validate class ID
            # ------------------------------------------------

            if not 0 <= class_id < len(CLASS_NAMES):
                invalid_lines.append(
                    {
                        "label": str(label_path),
                        "line": line_number,
                        "reason": (
                            f"Invalid class_id={class_id}, "
                            f"expected 0~{len(CLASS_NAMES) - 1}"
                        ),
                        "content": line,
                    }
                )
                continue

            # ------------------------------------------------
            # Validate bbox dimensions
            # ------------------------------------------------

            if box_width <= 0 or box_height <= 0:
                invalid_lines.append(
                    {
                        "label": str(label_path),
                        "line": line_number,
                        "reason": "width or height <= 0",
                        "content": line,
                    }
                )
                continue

            # Coordinates are normally in [0, 1].
            # We do not immediately reject slight rounding overflow,
            # because bbox clipping below can safely handle it.
            if (
                x_center < -0.01
                or x_center > 1.01
                or y_center < -0.01
                or y_center > 1.01
                or box_width > 1.01
                or box_height > 1.01
            ):
                invalid_lines.append(
                    {
                        "label": str(label_path),
                        "line": line_number,
                        "reason": "Suspicious normalized coordinates",
                        "content": line,
                    }
                )
                continue

            bbox = yolo_box_to_coco(
                x_center=x_center,
                y_center=y_center,
                box_width=box_width,
                box_height=box_height,
                image_width=image_width,
                image_height=image_height,
            )

            if bbox is None:
                invalid_lines.append(
                    {
                        "label": str(label_path),
                        "line": line_number,
                        "reason": "Invalid bbox after clipping",
                        "content": line,
                    }
                )
                continue

            x, y, w, h = bbox

            area = w * h

            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,

                    # Keep VisDrone/YOLO ID 0~9 directly
                    "category_id": class_id,

                    # COCO bbox format:
                    # [x_min, y_min, width, height]
                    "bbox": [
                        round(x, 4),
                        round(y, 4),
                        round(w, 4),
                        round(h, 4),
                    ],

                    "area": round(area, 4),

                    "iscrowd": 0,
                }
            )

            annotation_id += 1
            class_counter[class_id] += 1

    # --------------------------------------------------------
    # Save COCO JSON
    # --------------------------------------------------------

    with output_json.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            coco,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(f"Finished: {split}")
    print("=" * 80)

    print(f"Images             : {len(coco['images'])}")
    print(f"Annotations        : {len(coco['annotations'])}")
    print(f"Missing label files: {len(missing_labels)}")
    print(f"Empty label files  : {len(empty_labels)}")
    print(f"Invalid label lines: {len(invalid_lines)}")

    print()
    print("Class statistics:")

    for class_id, class_name in enumerate(CLASS_NAMES):
        print(
            f"  {class_id:2d} "
            f"{class_name:18s}: "
            f"{class_counter[class_id]}"
        )

    if missing_labels:
        print()
        print("[WARNING] First 10 images without label files:")

        for name in missing_labels[:10]:
            print(f"  {name}")

    if invalid_lines:
        print()
        print("[WARNING] First 10 invalid label lines:")

        for item in invalid_lines[:10]:
            print(
                f"  {item['label']}:{item['line']} "
                f"| {item['reason']} "
                f"| {item['content']}"
            )

    print()
    print(f"COCO JSON saved to:")
    print(output_json)
    print()

    return {
        "split": split,
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
        "missing_labels": len(missing_labels),
        "empty_labels": len(empty_labels),
        "invalid_lines": len(invalid_lines),
        "class_counter": dict(class_counter),
        "output": str(output_json),
    }


def main():

    parser = argparse.ArgumentParser(
        description="Convert VisDrone YOLO labels to COCO JSON"
    )

    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(
            "/home/ubuntu/WBA/RT-DETR-WBA/"
            "rtdetr_pytorch/datasets/visdrone2019"
        ),
        help="VisDrone dataset root directory",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help=(
            "Dataset splits to convert. "
            "Default: train val"
        ),
    )

    args = parser.parse_args()

    print()
    print("Dataset root:")
    print(args.base_dir)
    print()

    summaries = []

    for split in args.splits:
        summary = convert_split(
            base_dir=args.base_dir,
            split=split,
        )
        summaries.append(summary)

    print("=" * 80)
    print("ALL DONE")
    print("=" * 80)

    for item in summaries:
        print(
            f"{item['split']:8s} | "
            f"images={item['images']:6d} | "
            f"annotations={item['annotations']:8d} | "
            f"invalid={item['invalid_lines']:5d}"
        )


if __name__ == "__main__":
    main()