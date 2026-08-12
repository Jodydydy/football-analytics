# Engineering decisions

Each entry: what was decided, what the alternatives were, and what breaks if you choose
differently. This file is the reason the repository is worth reading — the code shows
*what*, this shows *why*.

> Заполняется по мере переноса. Пустой раздел = кусок, который ты ещё не разобрал.
> **Правило: нельзя перенести файл, не дописав сюда решения, которые в нём зашиты.**

---

## D1. MOT17 half-split instead of a held-out sequence split

**Decided:** first 50% of frames of each sequence to train, second 50% to validation.

**Alternative:** hold out whole sequences.

**Why:** MOT17 publishes ground truth only for the training sequences. A sequence-level
split would cost half the already small dataset, and the remaining sequences differ so
much in camera angle and density that the validation estimate would be dominated by
which sequences happened to be held out.

**What it costs:** the split is in-domain — same scenes, same people in both halves.
mAP@50 0.903 is a fit-quality number, not a generalisation estimate. Transfer is
measured separately on CAVIAR / ChokePoint / Edinburgh.

**Precedent:** this is the standard `train_half` / `val_half` protocol used by ByteTrack
and FairMOT, not an ad-hoc choice.

---

## D2. Reporting mAP@50 rather than mAP@50-95

**Decided:** headline metric is mAP@50; mAP@50-95 reported alongside.

**Why:** the detector feeds a tracker that scores line crossings from box centroids.
Localisation tightness beyond IoU 0.5 does not change whether an event fires, so
mAP@50-95 penalises a property the downstream task does not consume.

**What it costs:** mAP@50-95 = 0.566 shows the boxes are loose. If the pipeline ever
needs precise geometry — distance estimation, homography to a floor plan, occlusion
reasoning — this number becomes the relevant one.

---

## D3. Training a demographics model instead of using InsightFace `buffalo_l`

**Decided:** train a MobileNetV3-Large multi-task model from ImageNet weights.

**Alternative:** use pretrained `buffalo_l` genderage head.

**Why:** `buffalo_l` ships as ONNX without training code, so it cannot be fine-tuned to
the target domain. And on the actual benchmark it lost badly — gender 0.811 vs 0.936,
age (3 groups) 0.652 vs 0.869 — even after the comparison was tilted in its favour by
excluding its 2.6% detection failures from the denominator.

**What it costs:** a model to maintain, and licence entanglement from the training data
(see `DATASETS.md`).

**Honest caveat:** on clean frontal faces `buffalo_l` is still slightly ahead
(ChokePoint: 1.000 vs 0.964, n=28). The custom model's advantage is on varied and
imperfect faces, and it is consistently better on age in every domain tested.

---

## D4. Accepting 0.87 on 3-group age instead of chasing 0.9

**Decided:** freeze v2b at 0.869 and document what would be needed to go further.

**What was tried:** DLDL soft Gaussian age labels with KL loss, a dedicated 3-group
head, class-balanced sampling at two strengths (p = 0.5 and p = 0.7), EMA, 18 epochs.

**Why it stopped:** both sampler strengths landed at ≈0.87, a difference within standard
error (≈0.3 pp). The 50+ group has ~14k samples against ~68k for 20-49; oversampling
re-shows the same faces without adding information, and as the learning rate anneals the
model drifts back toward the majority group (50+ recall 0.764 → 0.725).

**Conclusion:** the ceiling is in the data, not the loss function. Moving it requires
more 50+ samples (AgeDB, IMDB-WIKI elderly, or own annotation).

**Why this is the right call:** continuing to tune would have produced a number that
does not generalise. Recording the ceiling and its cause is more useful than beating it
on the validation set.

---

## D5. Reporting a 3-group age segmentation, not 4 or 9

**Decided:** the product-facing output is three groups — 0-19 / 20-49 / 50+.

**Why:** measured, not assumed. Coarser buckets do **not** automatically give higher
accuracy — the errors concentrate on the 20-29 ↔ 30-39 boundary, so bucketing that
crosses it gains nothing. Measured alternatives: 3 groups 0.851, 2 groups 0.840,
4 groups 0.783 and 0.742 depending on cut points.

**Consequence:** three groups is the finest segmentation that stays reliable enough to
report to a client.

---

## D6. Parity check after ONNX export

**Decided:** every export is validated against the PyTorch model before use —
max|Δ| ≈ 1e-6 and argmax agreement on a fixed sample.

**Why:** silent divergence after export is a classic failure. Op-set differences,
normalisation baked in on one side only, or dynamic-axis mistakes produce a model that
runs fine and predicts differently.

---

## D7. Dropping empty and unusable inputs at ingestion

TODO — перенести решения из счётчиков и препроцессинга.

---

## D8. Line crossing vs zone dwell

TODO — почему для входа линия, а для выдачи и столов полигон с таймером; что
происходит с человеком, постоявшим в дверях; как определяется направление in/out.

---

## D9. Ground truth annotation

TODO — как размечался эталон, кем, какие правила для неоднозначных событий
(группа входит вместе, человек разворачивается на пороге), почему нет оценки
межаннотаторного согласия и что это значит для доверия к F1.

---

## D10. Not publishing client data

**Decided:** no frames, overlays, annotations or client name from the pilot footage.

**Why:** the footage is private CCTV from a live retail location containing identifiable
people. No NDA was signed, which makes the decision *more* conservative, not less —
there is no agreement defining what would have been permitted.

**Consequence:** the entry/exit F1 cannot be independently reproduced from this
repository. That is stated openly in `METRICS.md` rather than hidden behind a number.
