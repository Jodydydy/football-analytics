"""
Export the trained GenAgeNet checkpoint (best.pt) to ONNX, then verify
torch vs onnxruntime parity on a few val images.

Output: runs/mnv3_v1/mnv3_genage.onnx
  input  : "input"         float32 [N,3,224,224]  (ImageNet-normalized RGB)
  outputs: "gender_logits" [N,2]   (argmax: 0=male, 1=female)
           "age_logits"    [N,9]   (argmax -> bucket 0..8)
Age buckets: 0-2,3-9,10-19,20-29,30-39,40-49,50-59,60-69,70+

Usage:
  python export_onnx.py [--ckpt runs/mnv3_v1/best.pt] [--opset 17]
"""
import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from train import GenAgeNet, AGE_BUCKETS, IMAGENET_MEAN, IMAGENET_STD

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "mnv3_v1", "best.pt"))
    ap.add_argument("--manifest", default=os.path.join(HERE, "manifest.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "mnv3_v1", "mnv3_genage.onnx"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = GenAgeNet(n_age=9, pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval()

    dummy = torch.randn(1, 3, args.size, args.size)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["input"],
        output_names=["gender_logits", "age_logits"],
        dynamic_axes={"input": {0: "N"},
                      "gender_logits": {0: "N"},
                      "age_logits": {0: "N"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    sz = os.path.getsize(args.out) / 1e6
    print(f"exported -> {args.out}  ({sz:.1f} MB, opset {args.opset})")
    print(f"age_buckets = {AGE_BUCKETS}")

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
        tg, ta = model(batch)
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    og, oa = sess.run(["gender_logits", "age_logits"],
                      {"input": batch.numpy()})
    dg = float(np.abs(tg.numpy() - og).max())
    da = float(np.abs(ta.numpy() - oa).max())
    same_g = int((tg.argmax(1).numpy() == og.argmax(1)).sum())
    same_a = int((ta.argmax(1).numpy() == oa.argmax(1)).sum())
    print(f"\nparity (n={len(rows)}): max_abs_d_gender={dg:.2e}  max_abs_d_age={da:.2e}")
    print(f"argmax agreement: gender {same_g}/{len(rows)}  age {same_a}/{len(rows)}")
    assert dg < 1e-3 and da < 1e-3, "ONNX/torch mismatch too large"
    print("OK — ONNX matches torch.")


if __name__ == "__main__":
    main()
