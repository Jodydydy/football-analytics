"""
Export the v2 GenAgeNet2 checkpoint (3 heads) to ONNX, then verify torch vs
onnxruntime parity on a few val images.

v2 differs from v1: the model has a THIRD head (age3_head, 512->3) that predicts
the business 3-group split 0-19/20-49/50+ directly. That head is the production
age signal (age3_head metric); the 9-bucket head is kept for compatibility.

Output: runs/mnv3_v2b/mnv3_genage_v2.onnx
  input  : "input"         float32 [N,3,256,256]  (ImageNet-normalized RGB)
  outputs: "gender_logits" [N,2]   (argmax: 0=male, 1=female)
           "age_logits"    [N,9]   (argmax -> bucket 0..8)
           "age3_logits"   [N,3]   (argmax: 0=0-19, 1=20-49, 2=50+)  <-- production

Usage:
  python export_onnx_v2.py [--ckpt runs/mnv3_v2b/best.pt] [--opset 17] [--size 256]
"""
import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from train_v2 import GenAgeNet2
from train import AGE_BUCKETS, IMAGENET_MEAN, IMAGENET_STD

HERE = os.path.dirname(os.path.abspath(__file__))
AGE3_GROUPS = ["0-19", "20-49", "50+"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "mnv3_v2b", "best.pt"))
    ap.add_argument("--manifest", default=os.path.join(HERE, "manifest.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "mnv3_v2b", "mnv3_genage_v2.onnx"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"ckpt metrics: {ck.get('metrics')}")
    model = GenAgeNet2(pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval()

    dummy = torch.randn(1, 3, args.size, args.size)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["input"],
        output_names=["gender_logits", "age_logits", "age3_logits"],
        dynamic_axes={"input": {0: "N"},
                      "gender_logits": {0: "N"},
                      "age_logits": {0: "N"},
                      "age3_logits": {0: "N"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    sz = os.path.getsize(args.out) / 1e6
    print(f"exported -> {args.out}  ({sz:.1f} MB, opset {args.opset})")
    print(f"age_buckets = {AGE_BUCKETS}")
    print(f"age3_groups = {AGE3_GROUPS}")

    # ---- parity check on a few val images ----
    import onnxruntime as ort
    rows = []
    with open(args.manifest, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "val":
                rows.append(r["path"])
            if len(rows) >= 8:
                break
    tf = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    batch = torch.stack([tf(Image.open(p).convert("RGB")) for p in rows])
    with torch.no_grad():
        tg, ta, t3 = model(batch)
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    og, oa, o3 = sess.run(["gender_logits", "age_logits", "age3_logits"],
                          {"input": batch.numpy()})
    dg = float(np.abs(tg.numpy() - og).max())
    da = float(np.abs(ta.numpy() - oa).max())
    d3 = float(np.abs(t3.numpy() - o3).max())
    same_g = int((tg.argmax(1).numpy() == og.argmax(1)).sum())
    same_a = int((ta.argmax(1).numpy() == oa.argmax(1)).sum())
    same_3 = int((t3.argmax(1).numpy() == o3.argmax(1)).sum())
    print(f"\nparity (n={len(rows)}): max_abs_d gender={dg:.2e} age={da:.2e} age3={d3:.2e}")
    print(f"argmax agreement: gender {same_g}/{len(rows)}  "
          f"age {same_a}/{len(rows)}  age3 {same_3}/{len(rows)}")
    assert dg < 1e-3 and da < 1e-3 and d3 < 1e-3, "ONNX/torch mismatch too large"
    print("OK — ONNX matches torch.")


if __name__ == "__main__":
    main()
