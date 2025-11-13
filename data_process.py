#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build participants_data.json from raw CSVs under ./data.

Expected input files:
  - data/demographic_data.csv
  - data/daily_steps.csv
  - data/daily_activity.csv
  - data/sleep_stages.csv
  - data/sleep_classic.csv
  - data/mood.csv

Output:
  - participants_data.json

JSON structure per participant (example):

{
  "P311": {
    "Demographic": {
      "arm": "...",
      "cg_hours": "...",
      "age": "...",
      "gender": "...",
      "monthly_income": "...",
      "transplant_type": "...",
      "dyad_id": 1,
      "role": "Caregivers",
      "in_hospital_days": "OHSU patient"
    },
    "Daily steps": {
      "Day 0": 11287.0,
      "Day 4": 17519.0,
      ...
    },
    "Sedentary (minutes)": {
      "Day 0": 1159.0,
      ...
    },
    "Active minutes": {
      "Day 0": 278.0,
      ...
    },
    "Sleep duration (hours)": {
      "Day 0": 8.07,
      ...
    },
    "Sleep duration (classic, hours)": {
      "Day 0": 2.45,
      ...
    },
    "Mood": {
      "Day 0": 8.0,
      "Day 1": 8.0,
      ...
    }
  },
  ...
}
"""

import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
from huggingface_hub import snapshot_download

# --------------------
# Config: download dataset from Hugging Face and set DATA_DIR
# --------------------

HF_DATASET_ID = "nichenshun/roadmap-participants-data"  

# This will download (or reuse cached) snapshot of the dataset repo
# and return the local path to the repo root.
LOCAL_DATASET_DIR = Path(
    snapshot_download(
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        # revision="main", 
    )
)

# Your CSVs live under the "data" folder inside the dataset repo
DATA_DIR = LOCAL_DATASET_DIR / "rdmap_data"

DEMOGRAPHIC_CSV_PATH    = DATA_DIR / "demographic_data.csv"
DAILY_STEPS_CSV_PATH    = DATA_DIR / "daily_steps.csv"
DAILY_ACTIVITY_CSV_PATH = DATA_DIR / "daily_activity.csv"
SLEEP_STAGES_CSV_PATH   = DATA_DIR / "sleep_stages.csv"
SLEEP_CLASSIC_CSV_PATH  = DATA_DIR / "sleep_classic.csv"
MOOD_CSV_PATH           = DATA_DIR / "mood.csv"

OUTPUT_JSON_PATH        = "participants_data.json"

# If you want to clamp/sort within a day range:
MIN_DAY = 0
MAX_DAY = 120  # inclusive upper bound for sorting purposes


def safe_float(val: Any) -> Optional[float]:
    """Convert to float if possible, otherwise return None."""
    try:
        if pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None


def is_day_key(k: str) -> bool:
    """Return True if key looks like 'Day <int>' within [MIN_DAY, MAX_DAY]."""
    if not isinstance(k, str):
        return False
    k = k.strip()
    if not k.startswith("Day "):
        return False
    try:
        d = int(k.split(" ", 1)[1])
    except Exception:
        return False
    return MIN_DAY <= d <= MAX_DAY


def sort_day_map(dmap: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a mapping like {"Day 10": ..., "Day 2": ...}, return a new dict
    sorted by the numeric day ascending (Day 0..Day 120).
    """
    def day_num(k: str) -> int:
        try:
            return int(k.split(" ", 1)[1])
        except Exception:
            return 10**9  # push unknown keys to end if any

    # Keep only keys that look like 'Day <int>'; retain other keys at the end in original order (rare).
    day_items: List[Tuple[str, Any]] = [(k, v) for k, v in dmap.items() if is_day_key(k)]
    other_items: List[Tuple[str, Any]] = [(k, v) for k, v in dmap.items() if not is_day_key(k)]

    day_items_sorted = sorted(day_items, key=lambda kv: day_num(kv[0]))

    out: Dict[str, Any] = {}
    for k, v in day_items_sorted:
        out[k] = v
    # Append any non-conforming keys afterwards (shouldn't normally exist)
    for k, v in other_items:
        out[k] = v
    return out


def clamp_to_int_day(day_val: Any) -> Optional[int]:
    """Parse day (DaysFromTransplant) to int and clamp to [MIN_DAY, MAX_DAY]."""
    if pd.isna(day_val):
        return None
    try:
        d = int(day_val)
    except Exception:
        return None
    if d < MIN_DAY or d > MAX_DAY:
        # still allow beyond range if you prefer; here we clamp by rejecting outside
        return None
    return d


def main():

    # --------------------
    # 1. Build base dict from Demographic CSV
    # --------------------
    demo_path = DEMOGRAPHIC_CSV_PATH
    if not demo_path.exists():
        raise FileNotFoundError(f"Demographic CSV not found at: {demo_path.resolve()}")

    demo_df = pd.read_csv(demo_path)

    if "STUDY_PRTCPT_ID" not in demo_df.columns:
        raise ValueError("demographic_data.csv must contain 'STUDY_PRTCPT_ID' column.")

    participants: Dict[str, Any] = {}

    for _, row in demo_df.iterrows():
        pid = str(row["STUDY_PRTCPT_ID"])

        demo_dict = {
            "arm": row.get("arm"),
            "cg_hours": row.get("cg_hours"),
            "age": row.get("age"),
            "gender": row.get("gender"),
            "monthly_income": row.get("monthly_income"),
            "transplant_type": row.get("transplant_type"),
            "dyad_id": row.get("dyad_id"),
            "role": row.get("role"),
            "in_hospital_days": row.get("in_hospital_days"),
        }

        # Clean up NaN to None
        for k, v in list(demo_dict.items()):
            if isinstance(v, float) and math.isnan(v):
                demo_dict[k] = None

        participants[pid] = {
            "Demographic": demo_dict
        }

    print(f"[INFO] Loaded demographics for {len(participants)} participants.")

    # --------------------
    # 2. Add Daily steps from daily_steps.csv
    # --------------------
    steps_path = DAILY_STEPS_CSV_PATH
    if steps_path.exists():
        steps_df = pd.read_csv(steps_path)

        required_cols = {"STUDY_PRTCPT_ID", "DaysFromTransplant", "total_steps"}
        if not required_cols.issubset(set(steps_df.columns)):
            print(f"[WARN] daily_steps.csv missing one of columns {required_cols}, skipping Daily steps.")
        else:
            for _, row in steps_df.iterrows():
                pid = str(row["STUDY_PRTCPT_ID"])
                day_int = clamp_to_int_day(row["DaysFromTransplant"])
                steps = safe_float(row["total_steps"])

                if day_int is None or steps is None:
                    continue

                day_key = f"Day {day_int}"

                participants.setdefault(pid, {})
                participants[pid].setdefault("Daily steps", {})
                participants[pid]["Daily steps"][day_key] = steps

            print("[INFO] Added Daily steps information.")
    else:
        print(f"[WARN] daily_steps.csv not found at {steps_path.resolve()}, skipping Daily steps.")

    # --------------------
    # 3. Add Sedentary & Active minutes from daily_activity.csv
    # --------------------
    activity_path = DAILY_ACTIVITY_CSV_PATH
    if activity_path.exists():
        act_df = pd.read_csv(activity_path)

        required_cols = {"STUDY_PRTCPT_ID", "DaysFromTransplant", "sedentary", "total_active_minutes"}
        if not required_cols.issubset(set(act_df.columns)):
            print(f"[WARN] daily_activity.csv missing one of columns {required_cols}, skipping activity.")
        else:
            for _, row in act_df.iterrows():
                pid = str(row["STUDY_PRTCPT_ID"])
                day_int = clamp_to_int_day(row["DaysFromTransplant"])
                sed = safe_float(row["sedentary"])
                active = safe_float(row["total_active_minutes"])

                if day_int is None:
                    continue
                day_key = f"Day {day_int}"

                pid_dict = participants.setdefault(pid, {})

                if sed is not None:
                    pid_dict.setdefault("Sedentary (minutes)", {})
                    pid_dict["Sedentary (minutes)"][day_key] = sed

                if active is not None:
                    pid_dict.setdefault("Active minutes", {})
                    pid_dict["Active minutes"][day_key] = active

            print("[INFO] Added Sedentary (minutes) and Active minutes information.")
    else:
        print(f"[WARN] daily_activity.csv not found at {activity_path.resolve()}, skipping activity data.")

    # --------------------
    # 4. Add Sleep stages as Sleep duration (hours) from sleep_stages.csv
    # --------------------
    stages_path = SLEEP_STAGES_CSV_PATH
    if stages_path.exists():
        stages_df = pd.read_csv(stages_path)

        required_cols = {"STUDY_PRTCPT_ID", "DaysFromTransplant", "sleep_duration"}
        if not required_cols.issubset(set(stages_df.columns)):
            print(f"[WARN] sleep_stages.csv missing one of columns {required_cols}, skipping sleep stages.")
        else:
            for _, row in stages_df.iterrows():
                pid = str(row["STUDY_PRTCPT_ID"])
                day_int = clamp_to_int_day(row["DaysFromTransplant"])
                dur = safe_float(row["sleep_duration"])

                if day_int is None or dur is None:
                    continue

                day_key = f"Day {day_int}"

                pid_dict = participants.setdefault(pid, {})
                pid_dict.setdefault("Sleep duration (hours)", {})
                pid_dict["Sleep duration (hours)"][day_key] = round(dur, 2)

            print("[INFO] Added Sleep duration (hours) from sleep_stages.csv.")
    else:
        print(f"[WARN] sleep_stages.csv not found at {stages_path.resolve()}, skipping sleep stages.")

    # --------------------
    # 5. Add Sleep classic as Sleep duration (classic, hours) from sleep_classic.csv
    # --------------------
    classic_path = SLEEP_CLASSIC_CSV_PATH
    if classic_path.exists():
        classic_df = pd.read_csv(classic_path)

        required_cols = {"STUDY_PRTCPT_ID", "DaysFromTransplant", "sleep_duration"}
        if not required_cols.issubset(set(classic_df.columns)):
            print(f"[WARN] sleep_classic.csv missing one of columns {required_cols}, skipping sleep classic.")
        else:
            for _, row in classic_df.iterrows():
                pid = str(row["STUDY_PRTCPT_ID"])
                day_int = clamp_to_int_day(row["DaysFromTransplant"])
                dur = safe_float(row["sleep_duration"])

                if day_int is None or dur is None:
                    continue

                day_key = f"Day {day_int}"

                pid_dict = participants.setdefault(pid, {})
                pid_dict.setdefault("Sleep duration (classic, hours)", {})
                pid_dict["Sleep duration (classic, hours)"][day_key] = round(dur, 2)

            print("[INFO] Added Sleep duration (classic, hours) from sleep_classic.csv.")
    else:
        print(f"[WARN] sleep_classic.csv not found at {classic_path.resolve()}, skipping sleep classic.")

    # --------------------
    # 6. Add Mood from mood.csv
    # --------------------
    mood_path = MOOD_CSV_PATH
    if mood_path.exists():
        mood_df = pd.read_csv(mood_path)

        required_cols = {"STUDY_PRTCPT_ID", "DaysFromTransplant", "MOOD"}
        if not required_cols.issubset(set(mood_df.columns)):
            print(f"[WARN] mood.csv missing one of columns {required_cols}, skipping Mood.")
        else:
            for _, row in mood_df.iterrows():
                pid = str(row["STUDY_PRTCPT_ID"])
                day_int = clamp_to_int_day(row["DaysFromTransplant"])
                mood_val = safe_float(row["MOOD"])

                if day_int is None or mood_val is None:
                    continue

                day_key = f"Day {day_int}"

                pid_dict = participants.setdefault(pid, {})
                pid_dict.setdefault("Mood", {})
                pid_dict["Mood"][day_key] = mood_val

            print("[INFO] Added Mood information.")
    else:
        print(f"[WARN] mood.csv not found at {mood_path.resolve()}, skipping Mood.")

    # --------------------
    # 7. Reorder Day-* maps in ascending day order
    # --------------------
    DAY_SECTIONS = [
        "Daily steps",
        "Sedentary (minutes)",
        "Active minutes",
        "Sleep duration (hours)",
        "Sleep duration (classic, hours)",
        "Mood",
    ]

    for pid, node in participants.items():
        if not isinstance(node, dict):
            continue
        for section in DAY_SECTIONS:
            if section in node and isinstance(node[section], dict):
                node[section] = sort_day_map(node[section])

    # --------------------
    # 8. Save to JSON
    # --------------------
    out_path = Path(OUTPUT_JSON_PATH)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(participants, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote participants_data.json to {out_path.resolve()}")


if __name__ == "__main__":
    main()

