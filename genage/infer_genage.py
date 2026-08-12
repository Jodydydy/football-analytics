"""
Reusable gender/age inference on top of the exported ONNX model. Defaults to
the v2 model (runs/mnv3_v2b/mnv3_genage_v2.onnx, 256px, 3 heads) which has a
dedicated 3-group age head (age3_logits) — the production age signal
(age3_head acc ~0.87 vs collapsing the 9-bucket head). Backward compatible
with the v1 2-head model (auto-detected from ONNX outputs; falls back to
collapsing the 9-bucket head for age_group).

Intended for a FRONTAL entrance camera (faces visible) — NOT for top-down/
fish-eye (see project memory).

Three layers:
  1. GenAgePredictor  — ONNX face-crop -> {sex, age_bucket, age_group, probs}
  2. FaceGenAge       — full frame -> detect faces (SCRFD) -> predict each
  3. track + aggregate (video) — IOU tracker + weighted majority vote per person

Demo (per-track aggregation over a video, frontal camera):
  python infer_genage.py --video path/to/clip.mp4 --out-dir genage_demo --every 3
Single image:
  python infer_genage.py --image path/to/face_or_frame.jpg
Use the legacy v1 model:
  python infer_genage.py --image f.jpg --onnx runs/mnv3_v1/mnv3_genage.onnx --size 224
"""
import argparse
import csv
import os

import cv2
import numpy as np

AGE_BUCKETS = ["0-2", "3-9", "10-19", "20-29", "30-39",
               "40-49", "50-59", "60-69", "70+"]
# 3 business groups (recommended segmentation, ~0.85 acc): 0-19 / 20-49 / 50+
AGE_GROUPS = ["0-19", "20-49", "50+"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ONNX = os.path.join(HERE, "runs", "mnv3_v2b", "mnv3_genage_v2.onnx")
DEFAULT_SIZE = 256   # v2 model input; pass --size 224 for the legacy v1 model


def bucket_to_group(b):
    return 0 if b <= 2 else (1 if b <= 5 else 2)


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class GenAgePredictor:
    """ONNX gender/age on a pre-cropped face (BGR, any size)."""

    def __init__(self, onnx_path=DEFAULT_ONNX, providers=None, size=DEFAULT_SIZE):
        import onnxruntime as ort
        if providers is None:
            avail = ort.get_available_providers()
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if "CUDAExecutionProvider" in avail
                         else ["CPUExecutionProvider"])
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.size = size
        out_names = [o.name for o in self.sess.get_outputs()]
        # v2 has a dedicated 3-group head; v1 only gender+9-bucket age
        self.has_age3 = "age3_logits" in out_names
        self.out_names = (["gender_logits", "age_logits", "age3_logits"]
                          if self.has_age3 else ["gender_logits", "age_logits"])

    def _preprocess(self, face_bgr):
        img = cv2.resize(face_bgr, (self.size, self.size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return img.transpose(2, 0, 1)  # CHW

    def predict_batch(self, faces_bgr):
        x = np.stack([self._preprocess(f) for f in faces_bgr]).astype(np.float32)
        res = self.sess.run(self.out_names, {"input": x})
        gp, ap = _softmax(res[0]), _softmax(res[1])
        a3p = _softmax(res[2]) if self.has_age3 else None
        out = []
        for i in range(len(faces_bgr)):
            gi = int(gp[i].argmax())
            bi = int(ap[i].argmax())
            # production age_group: dedicated 3-head if present, else collapse 9-head
            if a3p is not None:
                gri = int(a3p[i].argmax())
                gr_probs = a3p[i].tolist()
            else:
                gri = bucket_to_group(bi)
                gr_probs = None
            out.append({
                "sex": "M" if gi == 0 else "F",
                "gender_prob": float(gp[i][gi]),
                "age_bucket": bi,
                "age_bucket_label": AGE_BUCKETS[bi],
                "age_group": gri,
                "age_group_label": AGE_GROUPS[gri],
                "age_probs": ap[i].tolist(),
                "age_group_probs": gr_probs,
            })
        return out

    def predict(self, face_bgr):
        return self.predict_batch([face_bgr])[0]


class FaceGenAge:
    """Full-frame: SCRFD face detection (insightface) + GenAgePredictor."""

    def __init__(self, onnx_path=DEFAULT_ONNX, det_size=640, margin=0.25,
                 providers=None, size=DEFAULT_SIZE):
        from insightface.app import FaceAnalysis
        import onnxruntime as ort
        avail = ort.get_available_providers()
        prov = providers or (["CUDAExecutionProvider", "CPUExecutionProvider"]
                             if "CUDAExecutionProvider" in avail
                             else ["CPUExecutionProvider"])
        ctx = 0 if "CUDAExecutionProvider" in avail else -1
        self.det = FaceAnalysis(name="buffalo_l", providers=prov,
                                allowed_modules=["detection"])
        self.det.prepare(ctx_id=ctx, det_size=(det_size, det_size))
        self.pred = GenAgePredictor(onnx_path, providers=prov, size=size)
        self.margin = margin

    def _crop(self, frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        mx, my = int(bw * self.margin), int(bh * self.margin)
        x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
        x2, y2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def detect(self, frame):
        faces = self.det.get(frame)
        results = []
        crops, boxes, scores = [], [], []
        for f in faces:
            bb = [int(v) for v in f.bbox]
            c = self._crop(frame, bb)
            if c is None or c.size == 0:
                continue
            crops.append(c); boxes.append(bb); scores.append(float(f.det_score))
        if not crops:
            return results
        preds = self.pred.predict_batch(crops)
        for bb, sc, p in zip(boxes, scores, preds):
            p = dict(p); p["bbox"] = bb; p["det_score"] = sc
            results.append(p)
        return results


# ---------- video: simple IOU tracker + weighted majority vote ----------
def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def run_video(video, onnx_path, out_dir, every=3, det_size=640,
              iou_thr=0.3, min_frames=3, size=DEFAULT_SIZE):
    os.makedirs(out_dir, exist_ok=True)
    engine = FaceGenAge(onnx_path, det_size=det_size, size=size)
    cap = cv2.VideoCapture(video)
    tracks = {}   # tid -> dict(bbox, votes_M, votes_F, age_w(9), w, n, last)
    next_id = 0
    fidx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx % every != 0:
            fidx += 1
            continue
        dets = engine.detect(frame)
        used = set()
        for d in dets:
            bb = d["bbox"]
            # match to existing track by IOU
            best, best_iou = -1, iou_thr
            for tid, t in tracks.items():
                if tid in used:
                    continue
                v = _iou(bb, t["bbox"])
                if v >= best_iou:
                    best, best_iou = tid, v
            if best < 0:
                best = next_id; next_id += 1
                tracks[best] = dict(bbox=bb, vM=0.0, vF=0.0, age_w=np.zeros(9),
                                    grp_w=np.zeros(3), w=0.0, n=0, last=fidx)
            used.add(best)
            t = tracks[best]
            area = (bb[2] - bb[0]) * (bb[3] - bb[1])
            wt = area * d["det_score"]          # bigger+confident faces weigh more
            if d["sex"] == "M":
                t["vM"] += wt
            else:
                t["vF"] += wt
            t["age_w"] += np.array(d["age_probs"]) * wt
            if d.get("age_group_probs") is not None:
                t["grp_w"] += np.array(d["age_group_probs"]) * wt
            t["w"] += wt; t["n"] += 1; t["bbox"] = bb; t["last"] = fidx
        fidx += 1
    cap.release()

    rows = []
    for tid, t in tracks.items():
        if t["n"] < min_frames:
            continue
        sex = "M" if t["vM"] >= t["vF"] else "F"
        ab = int(t["age_w"].argmax())
        # age_group from dedicated 3-head votes if available, else collapse bucket
        gri = int(t["grp_w"].argmax()) if t["grp_w"].sum() > 0 else bucket_to_group(ab)
        rows.append({
            "track_id": tid, "frames": t["n"],
            "sex": sex,
            "sex_conf": round(max(t["vM"], t["vF"]) / (t["vM"] + t["vF"] + 1e-9), 3),
            "age_bucket": AGE_BUCKETS[ab],
            "age_group": AGE_GROUPS[gri],
        })
    out_csv = os.path.join(out_dir, "tracks_genage.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["track_id", "frames", "sex",
                                          "sex_conf", "age_bucket", "age_group"])
        w.writeheader()
        w.writerows(rows)
    print(f"tracks: {len(rows)} (>= {min_frames} frames)  -> {out_csv}")
    for r in rows:
        print(f"  #{r['track_id']:>3} {r['frames']:>3}f  {r['sex']} "
              f"(conf {r['sex_conf']})  age {r['age_bucket']} [{r['age_group']}]")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE,
                    help="model input size (256 for v2, 224 for legacy v1)")
    ap.add_argument("--image", help="single image (face crop OR full frame)")
    ap.add_argument("--video", help="video for per-track aggregation")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "genage_demo"))
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--full-frame", action="store_true",
                    help="treat --image as full frame (run detector)")
    args = ap.parse_args()

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"cannot read {args.image}")
        if args.full_frame:
            eng = FaceGenAge(args.onnx, det_size=args.det_size, size=args.size)
            for d in eng.detect(img):
                print(f"  bbox={d['bbox']} det={d['det_score']:.2f} "
                      f"{d['sex']} ({d['gender_prob']:.2f})  "
                      f"age {d['age_bucket_label']} [{d['age_group_label']}]")
        else:
            p = GenAgePredictor(args.onnx, size=args.size).predict(img)
            print(f"  {p['sex']} ({p['gender_prob']:.2f})  "
                  f"age {p['age_bucket_label']} [{p['age_group_label']}]")
    elif args.video:
        run_video(args.video, args.onnx, args.out_dir, every=args.every,
                  det_size=args.det_size, size=args.size)
    else:
        raise SystemExit("provide --image or --video")


if __name__ == "__main__":
    main()
