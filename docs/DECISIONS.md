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

## D1b. Not shipping the fine-tuned detector

**Decided:** the pipeline runs on stock pretrained YOLO11l. The MOT17 fine-tune, despite
scoring mAP@50 0.903 on MOT17-val, was measured on the downstream task and rejected.

**The measurement:** line-crossing event F1 went from 0.825 to 0.839 on MOT17 (+1.4 pp)
and from 0.900 to 0.646 on CAVIAR (−25.4 pp). Twelve epochs, fully unfrozen, seven
sequences from one dataset — catastrophic forgetting.

**Why this is the important entry in this file:** the detector metric said the fine-tune
was a success. The product metric said it was a serious regression on any scene the
model had not seen. Both were true; only one of them mattered.

**What it changed in how everything else here is reported:** detector metrics and event
metrics are kept separate throughout, and no detector number is presented as evidence
that the system works. Optimising a proxy until it stops predicting the objective is a
failure mode that does not announce itself — it looks like progress right up to
deployment.

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

## D7. Line crossing for doors, zone dwell for counters

**Decided:** entrances are scored by a line crossing; pickup counters and tables by a
polygon with a dwell timer.

**Why they differ:** at a door the event *is* a transition — you are inside or outside,
and the interesting moment is the instant between. At a counter there is no transition
to detect; a customer and a passer-by occupy the same pixels, and the only thing that
separates them is **how long** they stay. A line at the counter would count everyone
walking past; a dwell zone at the door would miss anyone who walks through briskly.

**What breaks:** the dwell threshold is a business decision disguised as a parameter.
Too low and foot traffic cutting through the zone is counted as customers; too high and
quick pickups are missed. It is calibrated per site, and re-calibrated if the furniture
moves.

**Two accumulated failure modes, both fixed by state rather than by tuning:**

*Someone standing on the line.* Detector jitter moves the anchor point back and forth
across the line, emitting dozens of events from one person. Fixed with per-track dedup
in the counter (`SingleLineCounter.counted_in` / `counted_out`) and, in
post-processing, by cancelling opposite-direction pairs that land within a short window.

*Someone standing in the doorway.* A single line cannot distinguish a completed passage
from a hesitation. `TripwireCounter` requires both of a pair of parallel lines to be
crossed in the same direction within a frame budget. Cleaner, at the cost of missing
passages that are too fast (both lines in one frame) or too slow (gap exceeded).

**Direction** comes from the start→end orientation of the line, inherited from
`sv.LineZone`. Redrawing the line to fix reversed polarity is error-prone, so
`--invert-direction` exists to flip it at the log level instead.

---

## D8. Ground truth annotated by hand, by one person

**Decided:** all event-level ground truth was annotated manually with the project's own
tool (`eval/gt_marker.py`), by a single annotator.

**Rules applied:** a table counts as occupied when a person is physically seated or
standing at it for at least 3–5 seconds; consecutive people with gaps under 5 seconds
form one session; walking through the zone does not count.

**What it costs, stated plainly:** there is **no inter-annotator agreement estimate**.
On ambiguous events — a person loitering in the doorway, a group entering together,
someone turning back at the threshold — the label is one person's judgement, and the
F1 figures inherit that. The rules above were written down precisely because the
ambiguity is real, but written rules are not the same as a measured agreement rate.

**One documented exclusion:** in the table-occupancy clip, a third person seated at
`big_center` was fully occluded by a display stand. They were excluded from ground truth
rather than counted as a detector miss, on the grounds that the camera physically cannot
see them. This is defensible, and it means the metric measures the pipeline rather than
the installation — which is a different claim, and should be described as such.

---

## D9. Reporting the weak results too

**Decided:** `docs/METRICS.md` includes pickup detection at F1 0.647 (n=12, ±15 pp) and
labels the table-occupancy 0.990 as per-second agreement over three sessions rather than
event-level accuracy.

**Why:** a metrics table containing only the good numbers is not a metrics table. The
0.990 in particular reads as a stronger result than it is, and someone who quotes it
without the caveat will be unable to defend it the first time they are asked what the
sample size was.

---

## D10. Not publishing client data

**Decided:** no frames, overlays, annotations or client name from the pilot footage.

**Why:** the footage is private CCTV from a live retail location containing identifiable
people. No NDA was signed, which makes the decision *more* conservative, not less —
there is no agreement defining what would have been permitted.

**Consequence:** the entry/exit F1 cannot be independently reproduced from this
repository. That is stated openly in `METRICS.md` rather than hidden behind a number.
