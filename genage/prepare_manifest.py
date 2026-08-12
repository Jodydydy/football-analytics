"""
Build a unified gender/age manifest from FairFace + UTKFace.

Output: train_genage/manifest.csv with columns:
    path        absolute path to the (already aligned/cropped) face image
    gender      0 = male, 1 = female
    age_bucket  0..8  (FairFace's 9 age groups)
    source      'fairface' | 'utkface'
    split       'train' | 'val'

Both datasets already ship aligned/cropped face images, so no SCRFD step here.
UTKFace continuous age is mapped into FairFace's 9 buckets so the two label
spaces line up for joint training.
"""
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "datasets"))
FF = os.path.join(DATA, "fairface")
UTK = os.path.join(DATA, "utkface", "UTKFace")

# FairFace's 9 age groups, in order -> bucket index 0..8
AGE_BUCKETS = ["0-2", "3-9", "10-19", "20-29", "30-39",
               "40-49", "50-59", "60-69", "more than 70"]
FF_AGE_TO_IDX = {b: i for i, b in enumerate(AGE_BUCKETS)}
# upper edges used to map a continuous age -> bucket idx
BUCKET_UPPER = [2, 9, 19, 29, 39, 49, 59, 69, 200]


def age_to_bucket(age: int) -> int:
    for i, hi in enumerate(BUCKET_UPPER):
        if age <= hi:
            return i
    return len(BUCKET_UPPER) - 1


def load_fairface(rows):
    n = 0
    for split, csv_name in [("train", "fairface_label_train.csv"),
                            ("val", "fairface_label_val.csv")]:
        path = os.path.join(FF, csv_name)
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                img = os.path.join(FF, r["file"].replace("/", os.sep))
                if not os.path.exists(img):
                    continue
                gender = 0 if r["gender"] == "Male" else 1
                bucket = FF_AGE_TO_IDX[r["age"]]
                rows.append([img, gender, bucket, "fairface", split])
                n += 1
    return n


# UTKFace filename: [age]_[gender]_[race]_[date].jpg   gender 0=male 1=female
UTK_RE = re.compile(r"^(\d+)_(\d)_(\d)_.*\.jpg$", re.IGNORECASE)


def load_utkface(rows):
    n = skipped = 0
    if not os.path.isdir(UTK):
        print(f"WARN: UTKFace dir not found: {UTK}")
        return 0
    files = sorted(os.listdir(UTK))
    for i, fn in enumerate(files):
        m = UTK_RE.match(fn)
        if not m:
            skipped += 1
            continue
        age = int(m.group(1))
        gender = int(m.group(2))  # already 0=male 1=female
        if gender not in (0, 1) or age < 0 or age > 120:
            skipped += 1
            continue
        bucket = age_to_bucket(age)
        # deterministic 90/10 split by index
        split = "val" if (i % 10 == 0) else "train"
        rows.append([os.path.join(UTK, fn), gender, bucket, "utkface", split])
        n += 1
    if skipped:
        print(f"UTKFace: skipped {skipped} files with malformed names")
    return n


def main():
    rows = []
    n_ff = load_fairface(rows)
    n_utk = load_utkface(rows)
    out = os.path.join(HERE, "manifest.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "gender", "age_bucket", "source", "split"])
        w.writerows(rows)

    # summary
    from collections import Counter
    by_split = Counter(r[4] for r in rows)
    by_src = Counter(r[3] for r in rows)
    by_gender = Counter(r[1] for r in rows)
    by_age = Counter(r[2] for r in rows)
    print(f"\nWrote {len(rows)} rows -> {out}")
    print(f"  FairFace={n_ff}  UTKFace={n_utk}")
    print(f"  split: {dict(by_split)}")
    print(f"  source: {dict(by_src)}")
    print(f"  gender: male={by_gender[0]} female={by_gender[1]}")
    print("  age buckets:")
    for i, b in enumerate(AGE_BUCKETS):
        print(f"    {i} {b:>12}: {by_age[i]}")


if __name__ == "__main__":
    sys.exit(main())
