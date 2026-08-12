# Metrics

Every result here is reported with its evaluation protocol and sample size.
Numbers without a protocol are not results — they are decoration.

> **Status legend:** ✅ verified from run logs · 🔄 needs re-measuring before publication
> · ❌ removed, do not use

---

## 1. Person detection — MOT17 fine-tune

| Metric | Value |
|---|---|
| mAP@50 | **0.903** ✅ |
| mAP@50-95 | 0.566 ✅ |
| Precision | 0.915 |
| Recall | 0.838 |
| Val images | 2 659 |
| Val instances | 45 387 |

**Model:** YOLO11l fine-tuned on MOT17, single class (`person`).

**Split protocol.** Standard MOT17 half-split: the first 50% of frames of each training
sequence go to train, the second 50% to validation — the same convention as
`train_half.json` / `val_half.json` used by ByteTrack and FairMOT. MOT17 publishes
ground truth only for the training sequences, so no held-out sequence split is possible
without giving up half the data.

**What this number does and does not mean.** The split is *in-domain*: the same scenes,
cameras and people appear in both halves. mAP@50 0.903 measures how well the model fits
this domain after fine-tuning — it is **not** a generalisation estimate. Transfer to
unseen scenes is measured separately (§4).

**Why mAP@50 and not mAP@50-95.** The downstream consumer is a tracker that operates on
box centroids and a line-crossing test. Localisation tightness beyond IoU 0.5 does not
change whether a crossing is scored. mAP@50-95 = 0.566 is reported alongside so the
gap is visible rather than hidden.

**Annotation filter.** Only pedestrian class (`1`), `conf = 1`, visibility ≥ 0.1.

### ⚠️ This model was measured, and then not shipped

The fine-tune improves the detector metric and **degrades the product**. Measured on the
downstream task — line-crossing event F1, which is what the system actually delivers:

| | Baseline YOLO11l | Fine-tuned on MOT17 | Δ |
|---|---|---|---|
| MOT17 (in-domain) | 0.825 | 0.839 | **+1.4 pp** |
| CAVIAR (out-of-domain) | 0.900 | 0.646 | **−25.4 pp** |

Twelve epochs, fully unfrozen, on seven sequences from a single dataset. The result is
textbook **catastrophic forgetting**: the model specialised to MOT17's camera angles and
crowd densities and lost general person detection. mAP@50 = 0.903 on MOT17-val looked
excellent and was measuring the wrong thing.

**Decision: the fine-tuned weights were not used.** The pipeline runs on stock
pretrained weights.

This is the most useful negative result in the project, and the reason the detector
metric and the event metric are reported separately everywhere in this document. A
detector number is a proxy; the product metric is the one that decides. Anyone quoting
mAP@50 0.903 as evidence the system works has the causality backwards.

---

## 2. Event scoring on real footage

| Task | Metric | n | 95% CI (Wilson) | Source |
|---|---|---|---|---|
| Door events, **all** | F1 **0.921** | 89 events | P [0.85, 0.96] · R [0.85, 0.96] | 4 h private CCTV |
| Door events, customers only | F1 **0.957** | 24 events | [0.86, 0.99] | same footage |
| Queue detection (≥2 people) | F1 **0.875** | — | — | Café DVR |
| Table occupancy, per-second | F1 **0.990** | 3 sessions / ~9 min | — | Café DVR |
| Pickup hand-off, event-level | F1 **0.647** | 12 events | ±15 pp | Café DVR |
| Pickup, phone-shown | recall 12/21 = **0.57** | 21 | — | 4 h private CCTV |

### Which door number to quote

**0.921 over 89 events is the honest headline.** 0.957 is the same run restricted to
customer events only — a business-relevant subset, but a subset, and only 24 events
wide. Quoting the higher number without saying it is a 24-event subset invites exactly
one question, and there is no good answer to it.

At n=24 a single error moves F1 by several hundredths; the Wilson interval [0.86, 0.99]
spans the difference between "excellent" and "mediocre". This is a sanity check that the
pipeline survives real footage, not a production accuracy guarantee.

### The table-occupancy 0.990 is not what it looks like

It is **per-second occupancy agreement**, not event detection. The clip is 9:05 long
with three tables, so roughly 1 600 per-second samples — but only **three occupancy
sessions** in the ground truth, one of which covers the entire clip (`left_bar`,
occupied 00:00–09:05).

Per-second samples are massively autocorrelated: predicting one long session correctly
produces hundreds of consecutive "correct" seconds. The effective sample size is closer
to 3 than to 1 600, and the number should never be presented as event-level accuracy.

The ground truth also documents an **architectural blind spot**: a third person seated
at `big_center` was fully occluded by a display stand and therefore excluded from the
ground truth rather than counted as a detector miss. That decision is defensible — the
camera physically cannot see them — but it means the metric measures the pipeline, not
the installation.

### Pickup detection was reported honestly as weak

F1 0.647 on 12 events with a ±15 pp confidence interval was recorded in the project
notes as "слабый production claim" — a weak basis for a production claim. It is listed
here for the same reason: a metrics table that only contains the good results is not a
metrics table.

Redefining the metric from raw per-second intervals to visit level moved F1 from 0.18
to 1.0 on a 6-minute clip with a single visit. That is a genuine insight about metric
design — and simultaneously a sample of one, which is why it is not in the headline
table.

### Ground truth provenance

Annotated by hand, by one person, with the project's own tools (`eval/gt_marker.py`).
Single-annotator ground truth has **no inter-annotator agreement estimate**. On ambiguous
events — someone loitering in the doorway, a group entering together, a person turning
back at the threshold — that is a real source of bias, and it is not quantified here.

---

## 3. Demographics — custom multi-task model

**Training data:** FairFace (97 698) + UTKFace (23 705), single manifest of
**121 403 images**, train 108 078 / val 13 325. Age bucketed into 9 FairFace bins.

**Model:** MobileNetV3-Large (ImageNet pretrained), shared 512-d layer, two heads
(binary gender, 9-bin age). Version v2b adds a dedicated 3-group age head, DLDL soft
Gaussian age labels with KL loss, class-balanced sampling, EMA.

| Metric | v1 | **v2b (production)** |
|---|---|---|
| Gender | 0.930 | **0.936** ✅ |
| Age, 3 groups (0-19 / 20-49 / 50+) | 0.851 | **0.869** ✅ |
| Age, 9 bins exact | 0.555 | — |
| Age, ±1 bin | 0.944 | — |

### Baseline comparison

Pretrained InsightFace `buffalo_l` on the same validation set, **with the benchmark
tilted in its favour**: its 2.6% "no face detected" cases were excluded from the
denominator rather than counted as errors.

| Metric | buffalo_l | Ours | Δ |
|---|---|---|---|
| Gender | 0.811 | **0.936** | +12.5 pp |
| Age, 3 groups | 0.652 | **0.869** | +21.7 pp |
| Age, 9 bins | 0.262 | 0.555 | +29.3 pp |

### Ceiling analysis

Target was 0.9 on the 3-group age task. Two runs with different class-balanced
sampling strengths (p = 0.5 and p = 0.7) both settled at ≈0.87 — a difference within
standard error. Root cause is data, not optimisation: the 50+ group has ~14k samples
against ~68k for 20-49, so oversampling re-shows the same faces without adding
information, and the model drifts toward the majority group as the learning rate anneals
(50+ recall falls 0.764 → 0.725).

**Decision: accept 0.87 as the realistic ceiling on the available data** and record what
would be needed to move it (more 50+ samples — AgeDB, IMDB-WIKI elderly, or own
annotation), rather than tune the sampler further.

### ONNX export

Exported at opset 17 with dynamic batch, 13.9 MB. **Parity check against PyTorch:
max|Δ| ≈ 1e-6, argmax agreement 8/8.** Inputs `[N,3,256,256]` ImageNet-normalised RGB;
outputs `gender_logits[N,2]`, `age_logits[N,9]`, `age3_logits[N,3]`.

---

## 4. Cross-domain transfer

| Model | Gender accuracy | n | Dataset |
|---|---|---|---|
| buffalo_l (pretrained) | 1.000 | 28 | ChokePoint P1E_S1_C1 |
| **Ours (v2b)** | **0.964** | 28 | ChokePoint P1E_S1_C1 |

The custom model transfers to a frontal surveillance domain it was never trained on.
The pretrained baseline is slightly ahead here, which is consistent: `buffalo_l` is
strong on clean frontal faces, the custom model wins on varied and imperfect ones.

**n = 28.** One misclassification is 3.6%. Treat this as directional evidence, not a
measurement.

### Counting pipeline across public datasets

Stock pretrained detector, per-dataset line/zone geometry, event-level F1:

| Dataset | F1 | n (GT events) | Note |
|---|---|---|---|
| CAVIAR | **0.933** | 157 (TP 139 · FP 2 · FN 18) | Mall corridor, cooldown 120 |
| Mall | **0.901** | 156 | Density-counting dataset |
| MOT17, all 7 sequences | **0.889** | 342 | Same config across all sequences |
| Edinburgh Office, absence events | **1.000** | 16 | 5 days of footage |
| Edinburgh Forum | **0.9905** | 58 750 | ⚠️ per-frame, not per-event — see below |

**Edinburgh Forum 0.9905 is a per-frame figure** over 58 750 frames, and carries the same
autocorrelation problem as the table-occupancy number in §2: consecutive frames are not
independent samples. It is not comparable to the event-level F1 in the rows above it and
should not be quoted as though it were.

**No single configuration wins everywhere.** Per-sequence tuning of two or three
parameters (confidence, cooldown window, line placement) was needed on each dataset;
the production pattern that came out of this is an operator spending five to ten minutes
per installation, not a universal config. That is an honest description of the
deployment cost and worth saying out loud — it is the part vendors leave out.

**What did not work**, from the same experiments: lowering confidence to 0.10 did not
raise recall, because the detector was not filtering those people out — it physically
could not see them, being small or occluded. Raising `imgsz` to 1920 helped on sequences
with small distant subjects and actively hurt on low-resolution ones, where upscaling
added noise.

---

## 5. Throughput ✅

Measured with `scripts/benchmark_fps.py`. **RTX 4060, 640×360 source, imgsz 640,
batch 1, streaming (frame by frame), 200 frames after 20 warm-up frames.**

| Model | Precision | Detection only | **Full pipeline** |
|---|---|---|---|
| YOLO11l | FP32 | 56.8 fps | **47.1 fps** |
| YOLO11l | FP16 | 56.5 fps | **50.2 fps** |
| YOLO11m | FP32 | 68.4 fps | **63.6 fps** |
| YOLOv8n | FP32 | 97.2 fps | **79.0 fps** |

"Full pipeline" = decode → detect (letterbox + forward + NMS) → ByteTrack →
line-crossing scoring. "Detection only" is the inference call alone. Quoting the
latter as system throughput is the standard exaggeration in this domain; both are
listed here so the gap is visible.

### The bottleneck is not the GPU

Three observations point the same way:

1. YOLOv8n reaches **the same ~97 fps on CPU and on GPU**. If the GPU were the limit,
   moving to it would help.
2. FP16 buys only **+7%** on the full pipeline (47.1 → 50.2). Halving precision should
   pay far better if compute-bound.
3. The gap between detection-only and full pipeline (56.8 → 47.1, about 17%) is pure
   CPU work: tracking, polygon tests, per-frame bookkeeping.

The ceiling is **CPU-side**: letterbox preprocessing, NMS post-processing, and the
per-frame Python overhead of moving arrays between ultralytics, supervision and the
counters. Batching frames, moving preprocessing onto the GPU, or exporting to
TensorRT would be the next lever — a bigger GPU would not.

Decode is measured separately (~2000 fps into memory) and overlaps with the rest of
the work in a real run, so end-to-end throughput is higher than 1/(sum of stages).

### On the previously quoted "~140 fps"

Not reproducible with this pipeline on this hardware. 140 fps corresponds to 7.1 ms
per frame, which is plausible for the **pure inference line that ultralytics prints**
(it reports preprocess, inference and postprocess separately, and only the middle
number is that small). Reading the inference figure as pipeline throughput is an easy
mistake and exactly what interviewers probe for.

The honest claim is ~50 fps end-to-end with the large model, and the interesting part
is not the number but the profile: the pipeline is CPU-bound, not GPU-bound.

---

## 6. Removed claims

| Claim | Why removed |
|---|---|
| "Cross-dataset validation gave 0.99 on an independent site" | The 0.990 figure is table-occupancy F1 on a café DVR recording — a different task on a different recording. It was never a cross-dataset validation. |
| "~140 fps on RTX 4060" | Not reproducible. Replaced by a measured 47–50 fps end-to-end (§5), with the bottleneck identified. |
