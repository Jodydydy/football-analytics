"""Convert ChokePoint sequence ground truth to events.csv (frame,event).

ChokePoint sequence naming: P{1,2}{E,L}_S{1..5}_C{1..3}
  E = Entering (people walking into a space)
  L = Leaving (people walking out)

GT XML (per sequence) lists per-frame face landmarks per subject_id. We pick
the FIRST frame each subject_id appears in the sequence as that subject's
event frame, and tag it with the direction implied by the sequence name.

Usage:
    python tools/chokepoint_gt.py --xml P1E_S1_C1.xml --seq-name P1E_S1_C1 --out gt.csv

Or for a whole portal (concatenate events into one CSV with frame offsets):
    python tools/chokepoint_gt.py --xml-dir groundtruth/P1E/ --out gt.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

SEQ_RE = re.compile(r"P([12])([EL])_S([1-9])_C([1-3])", re.IGNORECASE)


def direction_of(seq_name: str) -> str:
    """E -> 'in', L -> 'out'. Raises if format unknown."""
    m = SEQ_RE.search(seq_name)
    if not m:
        raise ValueError(f"can't parse ChokePoint sequence name: {seq_name!r}")
    return "in" if m.group(2).upper() == "E" else "out"


def parse_gt_xml(xml_path: Path) -> dict[int, list[int]]:
    """subject_id -> sorted list of frame numbers where subject is present."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    subj_frames: dict[int, list[int]] = defaultdict(list)

    # ChokePoint XML structure: <frame number="N"><person id="K">...</person>...
    for frame in root.iter("frame"):
        fnum_attr = frame.attrib.get("number") or frame.attrib.get("id")
        if fnum_attr is None:
            continue
        try:
            fnum = int(fnum_attr)
        except ValueError:
            continue
        for person in frame.iter("person"):
            pid_attr = person.attrib.get("id")
            if pid_attr is None:
                continue
            try:
                pid = int(pid_attr)
            except ValueError:
                continue
            subj_frames[pid].append(fnum)

    for pid in subj_frames:
        subj_frames[pid].sort()
    return dict(subj_frames)


def emit_events_for_seq(xml_path: Path, seq_name: str | None) -> list[tuple[int, str]]:
    seq = seq_name or xml_path.stem
    direction = direction_of(seq)
    subj_frames = parse_gt_xml(xml_path)
    events = [(frames[0], direction) for frames in subj_frames.values() if frames]
    events.sort()
    return events


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--xml", type=Path, help="single sequence GT XML")
    g.add_argument("--xml-dir", type=Path, help="directory of sequence XMLs")
    ap.add_argument("--seq-name", default=None,
                    help="override sequence name (defaults to XML stem)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    all_events: list[tuple[int, str]] = []
    if args.xml:
        all_events = emit_events_for_seq(args.xml, args.seq_name)
        n_subj = len(parse_gt_xml(args.xml))
        print(f"{args.xml.name}: {n_subj} subjects -> {len(all_events)} events")
    else:
        offset = 0
        for xml_file in sorted(args.xml_dir.glob("*.xml")):
            evs = emit_events_for_seq(xml_file, None)
            shifted = [(f + offset, e) for f, e in evs]
            all_events.extend(shifted)
            # offset = next frame after the last in this file
            if evs:
                offset = max(f for f, _ in evs) + offset + 1
            print(f"  {xml_file.name}: {len(evs)} events (offset+={offset})")

    all_events.sort()
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "event"])
        for fn, ev in all_events:
            w.writerow([fn, ev])

    n_in = sum(1 for _, e in all_events if e == "in")
    n_out = sum(1 for _, e in all_events if e == "out")
    print(f"\ntotal: {len(all_events)} events ({n_in} in, {n_out} out) -> {args.out}")


if __name__ == "__main__":
    main()
