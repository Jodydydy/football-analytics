# Datasets

**No dataset is redistributed in this repository.** Everything below must be obtained
from its original source. Most are research-only; see the licence column before using
anything commercially.

Download helper: `tools/download_datasets.py` (TODO).

---

## Used for training

| Dataset | Size | What for | Licence | Source |
|---|---|---|---|---|
| MOT17 | ~4.5 GB | YOLO11l person-detection fine-tune | **Non-commercial** (CC-BY-NC) | https://motchallenge.net/data/MOT17/ |
| FairFace | ~97 700 imgs | Gender/age model, main source | **CC-BY 4.0 — commercial OK** | https://github.com/joojs/fairface |
| UTKFace | ~23 700 imgs | Gender/age model, additional | **Non-commercial research** | https://susanqq.github.io/UTKFace/ |

## Used for evaluation and transfer checks

| Dataset | Size | What for | Licence | Source |
|---|---|---|---|---|
| ChokePoint | ~3.4 GB | Portal/door scenes, demographics transfer test | Research only | https://arma.sourceforge.net/chokepoint/ |
| CAVIAR | ~0.32 GB | Surveillance, counting transfer | Research only | https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/ |
| Edinburgh Informatics Forum | ~6.2 GB | Overhead pedestrian tracking | Research only | https://homepages.inf.ed.ac.uk/rbf/FORUMTRACKING/ |
| Collective Activity | ~1.9 GB | Group behaviour | Research only | https://cvgl.stanford.edu/projects/collective/ |
| Mall | ~0.17 GB | Crowd counting | Research only | https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html |
| Beijing-BRT | ~0.05 GB | Station pedestrians | Research | https://github.com/XMU-smartdsp/Beijing-BRT-dataset |
| PETS 2009 | small | Crowd scenarios | Research | http://www.cvg.reading.ac.uk/PETS2009/ |
| HiEve | ~4.3 GB | Includes a `queuing` action class | Research only | http://humaninevents.org/ |
| CrowdHuman (head subset) | ~0.12 GB | Head-detection experiment | Research | https://www.crowdhuman.org/ |

---

## Not published

| Data | Why |
|---|---|
| Pilot CCTV from a marketplace pickup point (~2.2 GB) | **Client data.** Not published in any form — no frames, no overlays, no annotations, no client name. |
| Café DVR recordings | Own captures of a third-party public stream. Rights unclear; only derived metrics are reported. |
| Public-webcam captures | Same. |

Metrics computed on this footage are reported in `METRICS.md`, but the footage itself
never leaves local storage.

---

## Licence implications for the trained weights

The demographics model was trained on FairFace (CC-BY 4.0, commercial use permitted)
**and** UTKFace (non-commercial). The resulting weights therefore inherit the more
restrictive terms and are **not** available for commercial use.

Retraining on FairFace alone is the path to commercially clean weights — this was
checked during the project specifically because a commercial pilot was on the table.

The MOT17 fine-tuned detector is likewise non-commercial, since MOT17 is CC-BY-NC.
