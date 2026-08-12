# Footfall Analytics

Computer vision pipeline for retail footfall analytics: people counting at entrances,
queue length, table occupancy, staff presence, and demographic estimation from CCTV.

Built end to end as a solo R&D project — detector fine-tuning, tracking, event scoring,
an evaluation harness with hand-annotated ground truth, and a custom multi-task
demographics model exported to ONNX for edge inference.

Every number below is reported with its evaluation protocol, and with its sample size
wherever that size is small enough to matter. Several results come from small sets;
[`docs/METRICS.md`](docs/METRICS.md) states the caveats explicitly rather than burying
them.

---

## What it does

| Task | Approach |
|---|---|
| Entry / exit counting | YOLO11 → ByteTrack → line-crossing with per-track dedup |
| Tripwire counting | Paired parallel lines; both must be crossed in the same direction |
| Queue length | Zone polygon, occupancy threshold sustained over time |
| Transaction counting | Zone polygon + dwell timer, three counting modes |
| Staff absence | Inverse zone logic: alert when the area stays empty |
| Gender / age | Custom MobileNetV3-Large multi-task model, ONNX export |
| Heatmaps | Centroid logging for later density rendering |

## Results

| Task | Metric | Dataset |
|---|---|---|
| Person detection (fine-tune, **not shipped** — see below) | mAP@50 **0.903** · mAP@50-95 0.566 | MOT17, half-split, 2 659 images |
| Counting, cross-dataset | F1 **0.933** · **0.901** · **0.889** | CAVIAR (n=157) · Mall (n=156) · MOT17 (n=342) |
| Door events, all | F1 **0.921**, 95% CI [0.85, 0.96] | 4 h private CCTV, 89 GT events |
| Door events, customers only | F1 **0.957** | same footage, 24-event subset |
| Queue detection | F1 **0.875** | Café DVR recording |
| Table occupancy, per-second | F1 **0.990** | Café DVR, 3 sessions over 9 min |
| Gender classification | **0.936** | FairFace + UTKFace, n=13 325 |
| Age, 3 groups | **0.869** | FairFace + UTKFace, n=13 325 |
| Gender, cross-domain | **0.964** | ChokePoint, n=28 |
| End-to-end throughput | **47–50 fps** (YOLO11l, 640 px, batch 1) | RTX 4060 |

The demographics model beats pretrained InsightFace `buffalo_l` by 12 points on gender
and 20 on 3-group age, on a benchmark deliberately tilted in the baseline's favour.
Details and the ceiling analysis: [`docs/GENAGE_REPORT.md`](docs/GENAGE_REPORT.md).

The throughput profile is more interesting than the number: the pipeline is
**CPU-bound, not GPU-bound** — see [`docs/METRICS.md`](docs/METRICS.md) §5 for the
three measurements that show it.

**The most useful result here is a negative one.** Fine-tuning the detector on MOT17
raised mAP@50 to 0.903 and raised downstream counting F1 by 1.4 points on MOT17 — while
dropping it by **25.4 points on CAVIAR** (0.900 → 0.646). Catastrophic forgetting: the
model specialised to one dataset's camera angles and lost general person detection. The
fine-tuned weights were measured and **not shipped**; the pipeline runs on stock
pretrained weights. A detector metric is a proxy, and this one stopped predicting the
objective while still going up.

Two more caveats worth reading before quoting anything from that table. The 0.990 on table
occupancy is **per-second occupancy agreement over three sessions**, not event-level
accuracy — the effective sample size is far smaller than it looks. And the 0.957 on
doors is a 24-event customer-only subset of the 89-event run that scored 0.921. Both
are unpacked in [`docs/METRICS.md`](docs/METRICS.md) §2, along with a pickup-detection
result that came out weak (F1 0.647, n=12) and is reported anyway.

## Install

```bash
git clone https://github.com/jodydydy/footfall-analytics
cd footfall-analytics

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

For GPU inference install a CUDA build of PyTorch first, following the selector at
pytorch.org — the plain `pip install torch` wheel is CPU-only.

Baseline detector weights are downloaded by ultralytics on first use.

Weights trained in this project — the MOT17 fine-tuned detector and the demographics
model — are **not included**: model files do not belong in git, and their licences are
more restrictive than the code (see below). Training them from scratch is reproducible
from the scripts here; the datasets have to be obtained separately.

## Quick start

Place the counting line and check it on a still frame before processing hours of video:

```bash
python -m footfall.cli --source video.mp4 --line 0.55 --preview-line
```

`preview_line.jpg` shows the line, a green dot at the start point and a blue one at the
end. The start→end direction sets IN/OUT polarity — swap the endpoints, or pass
`--invert-direction`, if the counters come out reversed.

Then run:

```bash
python -m footfall.cli --source video.mp4 --line 0.55 --show \
    --cancel-window 15 --per-track-cooldown 60 --log events.csv
```

Zone-dwell mode, for counting transactions at a counter:

```bash
python -m footfall.cli --source video.mp4 --zone counter.json \
    --dwell 2.5 --count-mode hybrid-max --show
```

Live RTSP camera:

```bash
python -m footfall.cli --source "rtsp://user:pass@10.0.0.10:554/Streaming/Channels/101" \
    --line 0.5 --log /var/log/footfall/events.csv
```

Events are appended to a CSV, one row per event:

```
frame,timestamp,event,track_id
114,2026-08-12T20:47:08,in,1
260,2026-08-12T20:47:09,out,58
```

`python -m footfall.cli --help` lists every option, grouped by purpose.

## Benchmarking

```bash
python scripts/benchmark_fps.py --source video.mp4 --model yolo11l.pt \
    --device 0 --imgsz 640 --frames 200
```

Reports decode, detection, tracking and full-pipeline throughput separately, together
with the configuration that produced them — device, model, resolution, precision and
batch size. Quoting a detection-only figure as system throughput is the most common
exaggeration in this domain, so both are always printed.

## Reproducing the detection result

```bash
# MOT17 -> YOLO format, standard half-split (first 50% of each sequence to train)
python tools/mot_to_yolo.py --mot-root datasets/mot17/MOT17 --out datasets/mot17_yolo

# fine-tune
yolo detect train data=datasets/mot17_yolo/mot17.yaml model=yolo11l.pt imgsz=640

# validate — this is what produces the mAP figures quoted above
yolo val data=datasets/mot17_yolo/mot17.yaml model=runs/detect/train/weights/best.pt
```

Ground truth for event-level evaluation on other datasets is generated by
`tools/caviar_gt.py`, `tools/chokepoint_gt.py`, `tools/collective_gt.py` and
`tools/mot_to_crossing_gt.py`, then scored with `eval/compare_gt_pred.py` or
`eval/eval_door_events.py`.

`tools/detector_eval.py` is a separate thing: detector-only evaluation against
density-counting datasets (Mall, Beijing-BRT), where ground truth is a head-count per
frame rather than boxes.

Datasets are not redistributed here — see [`docs/DATASETS.md`](docs/DATASETS.md) for
sources and licences.

## Project layout

```
footfall/
  detect.py       YOLO wrapper: model loaded once, called per frame
  track.py        ByteTrack wrapper
  sources.py      video file / RTSP / webcam / folder of frames
  counters.py     line, tripwire pair, zone dwell, zone absence
  ensemble.py     head detections merged into person detections
  zones.py        polygon and line geometry, line preview
  postprocess.py  event-log cleanup: round-trip cancel, per-track cooldown
  pipeline.py     assembly of the above
  cli.py          command-line interface
genage/           demographics: manifest, training, evaluation, ONNX export, inference
eval/             event-level evaluation and ground-truth annotation tools
tools/            dataset conversion, detector evaluation, zone/line pickers
scripts/          throughput benchmarking
docs/             metrics, datasets, engineering decisions, demographics report
```

## Engineering decisions

[`docs/DECISIONS.md`](docs/DECISIONS.md) records what was chosen, what the alternatives
were, and what breaks if you choose differently — the half-split protocol and its
in-domain limitation, why mAP@50 is the headline metric here, why a custom demographics
model rather than a pretrained one, why 0.87 on age was accepted as a data ceiling
instead of tuned further, and why client footage is not published.

## Data and privacy

This repository contains **no video, no frames and no annotations**. The pilot ran on
private CCTV from a marketplace pickup point; that footage is not published and the
client is not named. Public datasets must be obtained from their original sources.

## Licence

Code: MIT (see `LICENSE`).

Trained weights are **not** MIT — they derive from datasets with mixed licences
(FairFace CC-BY 4.0, UTKFace and MOT17 non-commercial). Retraining the demographics
model on FairFace alone is the path to commercially usable weights; this was checked
during the project because a commercial pilot was on the table.

## Status

Solo R&D project, April–June 2026. Stopped after the demographics model was frozen.
Not maintained.
