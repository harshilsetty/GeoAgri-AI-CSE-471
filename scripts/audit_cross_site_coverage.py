"""Audit cross-site data coverage across Hampi, Khajuraho, Nalanda, and Pattadakal.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter

def main():
    v3_root = Path("datasets/geoagri_obb_v3")
    names = {0: "boulders", 1: "others", 2: "ruins", 3: "structures", 4: "vegetation"}

    site_data = defaultdict(lambda: {
        "scenes": set(),
        "images": 0,
        "splits": Counter(),
        "class_instances": Counter()
    })

    for split in ["train", "val", "test"]:
        for img_p in (v3_root / "images" / split).glob("*"):
            site = img_p.name.split("_")[0].lower()
            if "_jpg.rf." in img_p.name:
                scene_id = img_p.name.split("_jpg.rf.")[0]
            elif "_png.rf." in img_p.name:
                scene_id = img_p.name.split("_png.rf.")[0]
            else:
                scene_id = img_p.stem

            lbl_p = v3_root / "labels" / split / (img_p.stem + ".txt")

            site_data[site]["scenes"].add(scene_id)
            site_data[site]["images"] += 1
            site_data[site]["splits"][split] += 1

            if lbl_p.exists():
                with open(lbl_p) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cid = int(parts[0])
                            site_data[site]["class_instances"][cid] += 1

    print("=== CROSS-SITE AUDIT SUMMARY ===")
    out_dict = {}
    for site, data in sorted(site_data.items()):
        print(f"Site: {site.upper()}")
        print(f"  Unique Scenes: {len(data['scenes'])}")
        print(f"  Total Images: {data['images']} (Splits: {dict(data['splits'])})")
        print("  Instances per class:")
        for cid in range(5):
            print(f"    {names[cid]:<12}: {data['class_instances'][cid]}")
        print()
        out_dict[site] = {
            "unique_scenes": len(data["scenes"]),
            "total_images": data["images"],
            "splits": dict(data["splits"]),
            "class_instances": {names[cid]: data["class_instances"][cid] for cid in range(5)}
        }

    with open("artifacts/agp12_cross_site_audit.json", "w") as f:
        json.dump(out_dict, f, indent=2)

if __name__ == "__main__":
    main()

