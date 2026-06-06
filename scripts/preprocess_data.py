"""
Dataset download and preprocessing helpers.

Run individual functions or the CLI to set up data directories.

Usage:
  python -m gdt.scripts.preprocess_data --setup-dirs --data-root data/
  python -m gdt.scripts.preprocess_data --verify --data-root data/
"""

from __future__ import annotations

import argparse
import json
import os


REQUIRED_DIRS = [
    "vqa",
    "coco/annotations",
    "coco/train2017",
    "coco/val2017",
    "visual_genome",
    "visual_genome/images",
]

REQUIRED_FILES = {
    "vqa": [
        "v2_mscoco_train2014_annotations.json",
        "v2_OpenEnded_mscoco_train2014_questions.json",
        "v2_mscoco_val2014_annotations.json",
        "v2_OpenEnded_mscoco_val2014_questions.json",
    ],
    "coco/annotations": [
        "captions_train2017.json",
        "captions_val2017.json",
        "instances_train2017.json",
        "instances_val2017.json",
    ],
    "visual_genome": [
        "region_descriptions.json",
        "relationships.json",
    ],
}


def setup_dirs(data_root: str) -> None:
    """Create all required data subdirectories."""
    for d in REQUIRED_DIRS:
        path = os.path.join(data_root, d)
        os.makedirs(path, exist_ok=True)
        print(f"  [OK] {path}")
    print(f"\nDirectories created under {data_root}")
    print("Download dataset files and place them in the directories above.")
    print("\nDataset sources:")
    print("  VQA v2:         https://visualqa.org/download.html")
    print("  COCO:           https://cocodataset.org/#download")
    print("  Visual Genome:  https://visualgenome.org/api/v0/api_home.html")


def verify_data(data_root: str) -> bool:
    """Check which required files are present."""
    all_ok = True
    for subdir, files in REQUIRED_FILES.items():
        for fname in files:
            path = os.path.join(data_root, subdir, fname)
            exists = os.path.exists(path)
            status = "OK" if exists else "MISSING"
            print(f"  [{status}] {path}")
            if not exists:
                all_ok = False
    return all_ok


def create_dummy_data(data_root: str) -> None:
    """
    Create minimal stub JSON files for unit-testing without real datasets.
    These contain valid structure but only 2-3 examples.
    """
    # VQA stub
    vqa_dir = os.path.join(data_root, "vqa")
    os.makedirs(vqa_dir, exist_ok=True)

    ann = {
        "annotations": [
            {
                "question_id": 1,
                "image_id": 1,
                "answers": [{"answer": "yes"}, {"answer": "yes"}],
            }
        ]
    }
    q = {
        "questions": [
            {"question_id": 1, "image_id": 1, "question": "Is this a cat?"}
        ]
    }
    with open(os.path.join(vqa_dir, "v2_mscoco_val2014_annotations.json"), "w") as f:
        json.dump(ann, f)
    with open(os.path.join(vqa_dir, "v2_OpenEnded_mscoco_val2014_questions.json"), "w") as f:
        json.dump(q, f)

    # COCO captions stub
    cap_dir = os.path.join(data_root, "coco", "annotations")
    os.makedirs(cap_dir, exist_ok=True)
    coco_cap = {
        "images": [{"id": 1, "file_name": "000000000001.jpg"}],
        "annotations": [{"id": 1, "image_id": 1, "caption": "A cat on a mat."}],
    }
    with open(os.path.join(cap_dir, "captions_val2017.json"), "w") as f:
        json.dump(coco_cap, f)

    # COCO detection stub
    coco_det = {
        "images": [{"id": 1, "file_name": "000000000001.jpg", "width": 640, "height": 480}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 100, 80]},
        ],
        "categories": [{"id": 1, "name": "cat"}],
    }
    with open(os.path.join(cap_dir, "instances_val2017.json"), "w") as f:
        json.dump(coco_det, f)

    # VG stub
    vg_dir = os.path.join(data_root, "visual_genome")
    os.makedirs(vg_dir, exist_ok=True)
    vg_regions = [{"id": 1, "regions": [{"phrase": "a cat on a mat", "region_id": 1}]}]
    with open(os.path.join(vg_dir, "region_descriptions.json"), "w") as f:
        json.dump(vg_regions, f)

    vg_rels = [
        {
            "id": 1,
            "relationships": [
                {
                    "relationship_id": 1,
                    "subject": {"object_id": 1, "name": "cat"},
                    "predicate": "on",
                    "object": {"object_id": 2, "name": "mat"},
                }
            ],
        }
    ]
    with open(os.path.join(vg_dir, "relationships.json"), "w") as f:
        json.dump(vg_rels, f)

    print(f"Stub data written to {data_root}")


def main() -> None:
    p = argparse.ArgumentParser(description="GDT data preprocessing")
    p.add_argument("--data-root", default="data")
    p.add_argument("--setup-dirs", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--create-dummy", action="store_true",
                   help="Create minimal stub data for testing")
    args = p.parse_args()

    if args.setup_dirs:
        setup_dirs(args.data_root)
    if args.verify:
        ok = verify_data(args.data_root)
        print("\nAll required files present." if ok else "\nSome files are missing.")
    if args.create_dummy:
        create_dummy_data(args.data_root)


if __name__ == "__main__":
    main()
