"""Merge per-image divide-and-conquer pseudo-mask JSONs into one COCO instances file.

    python -m video.scripts.merge_pseudo_coco --in-dir /tmp/davis_pseudo_json \
        --out datasets/davis_pseudo/annotations/train.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pattern", default="*.json",
                    help="glob within in-dir (e.g. 'p_*.json' for CascadePSP-refined only)")
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.in_dir) / args.pattern)))
    images, annotations, categories = [], [], None
    ann_id = 1
    for f in files:
        d = json.load(open(f))
        categories = categories or d.get("categories")
        img = d["image"]
        images.append(img)
        for a in d["annotations"]:
            a["id"] = ann_id
            a["image_id"] = img["id"]
            a["iscrowd"] = a.get("iscrowd", 0)
            annotations.append(a)
            ann_id += 1

    coco = {
        "info": {"description": "divide-and-conquer pseudo-masks"},
        "licenses": [{"id": 1, "name": "", "url": ""}],
        "categories": categories or [{"id": 1, "name": "fg", "supercategory": "fg"}],
        "images": images,
        "annotations": annotations,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(coco, open(out, "w"))
    print(f"merged {len(files)} files -> {out}  ({len(images)} images, {len(annotations)} annotations)")


if __name__ == "__main__":
    main()
