# Footfall Analytics

Computer vision pipeline for retail footfall analytics: people counting at entrances,
queue length, table occupancy, staff presence, and demographic estimation from CCTV.

Built end to end as a solo R&D project — detector fine-tuning, tracking, event scoring,
an evaluation harness with hand-annotated ground truth, and a custom multi-task
demographics model exported to ONNX for edge inference.

Every number below is reported with its sample size and protocol. Several come from
small evaluation sets; [`docs/METRICS.md`](docs/METRICS.md) states the caveats
explicitly rather than burying them.

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
| Person detection | mAP@50 **0.903** · mAP@50-95 0.566 | MOT17, half-split, 2 659 images |
| Entry/exit counting | F1 **0.957** (24 GT events) | 4 h private CCTV |
| Table occupancy | F1 **0.990** | Café DVR recording |
| Queue detection | F1 **0.875** | Café DVR recording |
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

## Install

```bash
git clone https://github.com/<user>/footfall-analytics
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

Detector weights are downloaded by ultralytics on first use. Trained weights for the
demographics model are published under Releases, not in the repository.

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
# MOT17 -> YOLO format, standard half-split
python tools/mot_to_yolo.py --mot-root datasets/mot17/MOT17 --out datasets/mot17_yolo

# fine-tune
yolo detect train data=datasets/mot17_yolo/mot17.yaml model=yolo11l.pt imgsz=640

# evaluate
python tools/detector_eval.py --weights runs/detect/train/weights/best.pt
```

Datasets are not redistributed here — see [`docs/DATASETS.md`](docs/DATASETS.md) for
sources and licences.

## Project layout

```
footfall/        detection, tracking, counters, geometry, post-processing, CLI
genage/          demographics: manifest, training, evaluation, ONNX export, inference
eval/            event-level evaluation and ground-truth annotation tools
tools/           dataset conversion, detector evaluation, zone/line pickers
scripts/         benchmarking
docs/            metrics, datasets, engineering decisions, demographics report
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
