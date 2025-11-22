#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate prompts for ROADMAP 2.0 LLM simulation with customized background wording,
and append pre-target-day averages for activity/sleep signals.

New in this version:
- After the 7-day Mood record block, insert a JSON summary of ALL mood history prior to the target day:
    {
      "Missing days prior to Day {target_day}": <int>,
      "Average mood on completed days prior to Day {target_day}": <float or "missing">
    }
  This sits between the 7-day mood record and the activity/sleep averages block.
- The activity/sleep averages block remains identical to the previous version.
- If target_day == 0, historical blocks are still skipped.

Other rules kept from prior versions:
- Background wording transforms for Caregivers/Patients and their partner blocks.
- Mood block shows a dense record for the last 7 days.
- Instruction line uses: "It has been {target_day} days since your transplant..."

Run:
    python prompts_gen_with_avgs.py
"""

import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import os
import statistics

# --------------------
# Config
# --------------------
DEMOGRAPHICS_JSON_PATH = "participants_data.json"   # input
OUTPUT_DIR             = "prompts_pre7days_activity"        # output dir

TARGET_DAYS   = [0, 20, 40, 60, 80, 100, 120]
N_PATIENTS    = 100
N_CAREGIVERS  = 100

MOOD_START_DAY   = 0
GLOBAL_MAX_DAY   = 120
MOOD_WINDOW_DAYS = 7   # Mood区块仅展示 [target_day - 7, target_day - 1]（若 target_day<7 自动裁剪）

RANDOM_SEED = 42

# --------------------
# Load participants JSON
# --------------------
in_path = Path(DEMOGRAPHICS_JSON_PATH)
if not in_path.exists():
    raise FileNotFoundError(f"Input JSON not found at: {in_path.resolve()}")

with in_path.open("r", encoding="utf-8") as f:
    participants: Dict[str, Any] = json.load(f)

if not isinstance(participants, dict):
    raise ValueError("Input JSON must be a dict keyed by STUDY_PRTCPT_ID.")

# --------------------
# Prepare dyad mappings and role-specific lists
# --------------------
dyad_to_patient_tx: Dict[str, Any] = {}
dyad_to_patient_hosp: Dict[str, Any] = {}
dyad_to_pids: Dict[str, List[str]] = defaultdict(list)

patient_ids: List[str] = []
caregiver_ids: List[str] = []

for pid, node in participants.items():
    if not isinstance(node, dict):
        continue
    demo = node.get("Demographic", {}) or {}
    if not isinstance(demo, dict):
        continue

    role   = demo.get("role")
    dyad_id = demo.get("dyad_id")
    tx     = demo.get("transplant_type")
    hosp   = demo.get("in_hospital_days")

    if role == "Patients":
        patient_ids.append(pid)
    elif role == "Caregivers":
        caregiver_ids.append(pid)

    if dyad_id is not None:
        dyad_to_pids[str(dyad_id)].append(pid)

    if role == "Patients" and dyad_id is not None:
        key = str(dyad_id)
        if tx:
            dyad_to_patient_tx.setdefault(key, tx)
        if hosp:
            dyad_to_patient_hosp.setdefault(key, hosp)

print(f"[INFO] Total patients: {len(patient_ids)}")
print(f"[INFO] Total caregivers: {len(caregiver_ids)}")

# --------------------
# Helpers
# --------------------
def _to_lower(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return str(s).lower()

def _is_int_like(x: Any) -> bool:
    try:
        int(str(x))
        return True
    except Exception:
        return False

def _parse_day_key(k: Any) -> Optional[int]:
    """
    Accept keys like 'Day N' or 'N' -> int N; return None if not parseable/out-of-range.
    """
    if k is None:
        return None
    s = str(k).strip()
    if s.lower().startswith("day "):
        s = s.split(" ", 1)[1].strip()
    try:
        d = int(s)
    except Exception:
        return None
    if d < MOOD_START_DAY or d > GLOBAL_MAX_DAY:
        return None
    return d

def _extract_series(node: Dict[str, Any], key: str) -> Dict[int, float]:
    """
    From participant node, fetch a sub-dict under `key` mapping 'Day N' -> value,
    parse days to int, keep finite floats only.
    """
    raw = node.get(key, {}) or {}
    out: Dict[int, float] = {}
    for k, v in raw.items():
        d = _parse_day_key(k)
        if d is None:
            continue
        try:
            val = float(v)
        except Exception:
            continue
        out[d] = val
    return out

def _avg_before_day(series: Dict[int, float], target_day: int) -> Optional[float]:
    """
    Average over all entries with day < target_day.
    Returns None if no valid entries before target_day.
    """
    if target_day <= 0:
        return None
    vals = [v for d, v in series.items() if d < target_day]
    if len(vals) == 0:
        return None
    try:
        return float(sum(vals) / len(vals))
    except Exception:
        try:
            return float(statistics.mean(vals))
        except Exception:
            return None

# --------------------
# Intros
# --------------------
def build_role_intro(demo: Dict[str, Any], include_decision_lead_in: bool) -> str:
    role = demo.get("role")
    if role == "Caregivers":
        role_word = "caregiver"
    elif role == "Patients":
        role_word = "patient"
    else:
        role_word = "participant"

    caregiver_intro = f"""You are a {role_word} participant in a dyad in the ROADMAP 2.0 mobile-health study. 

You are a caregiver for a loved one recovering from a hematopoietic cell transplantation (HCT) for a serious blood or immune disorder. You provide daily physical and emotional support, monitor for complications, manage medications and appointments, and balance these duties with your own daily work routine. 

Each day, you are asked to enter your mood on a 1-10 scale (1=worst, 10=best). 

The study seeks to understand links between sensor data and well-being and to support caregivers during and after transplant. 

Each time step in this simulation represents one day."""
    patient_intro = f"""You are a {role_word} participant in a dyad in the ROADMAP 2.0 mobile-health study. 

You are a patient recovering from a hematopoietic cell transplantation (HCT) to treat a blood-related disease such as leukemia or lymphoma. During recovery, you may experience fatigue, nausea, pain, sleep disturbances, and emotional stress, while relying on your care partner and medical team to help you regain strength and prevent complications like infections or graft-versus-host disease. 

Each day, you are asked to enter your mood on a 1-10 scale (1=worst, 10=best). 

The study seeks to understand links between sensor data and well-being and to support caregivers during and after transplant. 

Each time step in this simulation represents one day."""
    core = caregiver_intro if role == "Caregivers" else patient_intro

    if include_decision_lead_in:
        core += """

Below is your background and history with the program. Based on this information, decide whether you will complete today's mood survey."""
    return core.strip() + "\n\n"

# --------------------
# Background (own)
# --------------------
def build_own_background_bullets(demo: Dict[str, Any]) -> List[str]:
    age              = demo.get("age")
    gender           = demo.get("gender")
    monthly_income   = demo.get("monthly_income")
    arm              = demo.get("arm")
    role             = demo.get("role")
    transplant_type  = demo.get("transplant_type")
    in_hospital_days = demo.get("in_hospital_days")
    cg_hours         = demo.get("cg_hours")

    is_caregiver = (role == "Caregivers")
    is_patient   = (role == "Patients")

    lines: List[str] = []
    if age:
        lines.append(f"- You are in the {age} age group.")
    if gender:
        lines.append(f"- Your gender is {_to_lower(gender)}.")
    if monthly_income:
        lines.append(f"- Your monthly income is {monthly_income}.")
    if is_patient and arm:
        lines.append(f"- You are assigned to the {arm} arm of the study.")
    if is_patient and transplant_type:
        lines.append(f"- Your transplant type is {transplant_type}.")
    if in_hospital_days:
        if str(in_hospital_days) == "OHSU patient":
            if is_patient:
                lines.append("- You are an OHSU patient.")
        else:
            if is_patient and _is_int_like(in_hospital_days):
                lines.append(f"- You stayed in the hospital for {int(in_hospital_days)} days.")
    if cg_hours:
        if is_caregiver:
            lines.append(f"- You provide caregiving to your patient for {cg_hours} per week.")
        elif is_patient:
            lines.append(f"- You receive caregiving from your caregiver for {cg_hours} per week.")
        else:
            lines.append(f"- Caregiving hours recorded as {cg_hours} per week.")

    return lines

def build_own_background_block(demo: Dict[str, Any]) -> str:
    lines = ["- **Your Background:**"]
    lines.extend(build_own_background_bullets(demo))
    text = "\n".join(lines)
    return text if text.endswith("\n") else text + "\n"

# --------------------
# Background (partner)
# --------------------
def build_partner_background_block(own_demo: Dict[str, Any], partner_demo: Dict[str, Any]) -> str:
    partner_role = partner_demo.get("role")

    if partner_role == "Caregivers":
        title = "- **Your caregiver partner's background information**"
        noun = "caregiver"
        keep = {"age", "gender", "monthly_income"}
        lower_gender = True
        lower_income = False
        include_arm = include_tx = include_hosp = False
    else:
        title = "- **Your patient partner's background information**"
        noun = "patient"
        keep = {"age", "gender", "monthly_income", "arm", "transplant_type", "in_hospital_days"}
        lower_gender = True
        lower_income = True
        include_arm = include_tx = include_hosp = True

    age              = partner_demo.get("age")
    gender           = partner_demo.get("gender")
    monthly_income   = partner_demo.get("monthly_income")
    arm              = partner_demo.get("arm")
    transplant_type  = partner_demo.get("transplant_type")
    in_hospital_days = partner_demo.get("in_hospital_days")

    out: List[str] = [title]
    if "age" in keep and age:
        out.append(f"- Your {noun} is in the {age} age group.")
    if "gender" in keep and gender:
        g = _to_lower(gender) if lower_gender else gender
        out.append(f"- Your {noun}'s gender is {g}.")
    if "monthly_income" in keep and monthly_income:
        inc = _to_lower(monthly_income) if lower_income else monthly_income
        out.append(f"- Your {noun}'s monthly income is {inc}.")
    if "arm" in keep and include_arm and arm:
        out.append(f"- Your {noun} is assigned to the {arm} arm of the study.")
    if "transplant_type" in keep and include_tx and transplant_type:
        out.append(f"- Your {noun}'s transplant type is {transplant_type}.")
    if "in_hospital_days" in keep and include_hosp and in_hospital_days:
        if str(in_hospital_days) == "OHSU patient":
            out.append(f"- Your {noun} is an OHSU patient.")
        elif _is_int_like(in_hospital_days):
            out.append(f"- Your {noun} stayed in the hospital for {int(in_hospital_days)} days.")

    text = "\n".join(out)
    return text if text.endswith("\n") else text + "\n"

# --------------------
# Mood blocks
# --------------------
def build_mood_context_block_last7(node: Dict[str, Any], end_day: int) -> str:
    """
    Display dense Mood from max(0, end_day-7) .. end_day-1.
    Missing days filled with "missing".
    """
    mood_map_raw = node.get("Mood", {}) or {}
    day_to_val: Dict[int, float] = {}
    for k, v in mood_map_raw.items():
        d = _parse_day_key(k)
        if d is None:
            continue
        try:
            day_to_val[d] = float(v)
        except Exception:
            continue

    if end_day <= 0:
        return ""

    start = max(MOOD_START_DAY, end_day - MOOD_WINDOW_DAYS)
    stop  = max(MOOD_START_DAY, end_day)  # exclusive end

    dense: Dict[str, Any] = {}
    for d in range(start, stop):
        key = f"Day {d}"
        dense[key] = day_to_val.get(d, "missing")

    mood_json = json.dumps(dense, ensure_ascii=False, indent=2)

    lines = []
    lines.append("- **Context: past behavior:**")
    lines.append("Below is a record of your mood surveys from the previous 7 days (each entry represents one day):")
    lines.append(mood_json)

    text = "\n".join(lines)
    return text if text.endswith("\n") else text + "\n"

def build_mood_summary_stats_block(node: Dict[str, Any], end_day: int) -> str:
    """
    Summarize mood completion prior to end_day:
      - Missing count in the closed interval Day 0..Day (end_day-1)
      - Average mood over completed days in the same range (or "missing" if none)
    """
    if end_day <= 0:
        return ""

    mood_map_raw = node.get("Mood", {}) or {}
    day_to_val: Dict[int, float] = {}
    for k, v in mood_map_raw.items():
        d = _parse_day_key(k)
        if d is None:
            continue
        try:
            day_to_val[d] = float(v)
        except Exception:
            # non-numeric -> treat as not completed
            pass

    # Range: [0, end_day-1]
    total_days = max(0, end_day)
    completed_vals: List[float] = []
    present_days = set(day_to_val.keys())

    for d in range(0, end_day):
        if d in present_days:
            completed_vals.append(day_to_val[d])

    missing_count = total_days - len(completed_vals)
    avg_completed = round(sum(completed_vals) / len(completed_vals), 2) if completed_vals else "missing"

    payload = {
        f"Missing days prior to Day {end_day}": missing_count,
        f"Average mood on completed days prior to Day {end_day}": avg_completed
    }
    block_json = json.dumps(payload, ensure_ascii=False, indent=2)

    lines = []
    lines.append("Below is a summary of your historical mood completion prior to today:")
    lines.append(block_json)

    text = "\n".join(lines)
    return text if text.endswith("\n") else text + "\n"

# --------------------
# Activity/Sleep averages block (0..target_day-1)
# --------------------
def build_activity_sleep_avgs_block(node: Dict[str, Any], end_day: int) -> str:
    """
    Compute averages over ALL days < end_day for:
      - Daily steps
      - Sedentary (minutes)
      - Active minutes
      - Sleep duration (hours)
    Output a JSON block; any missing -> "missing".
    """
    steps_series = _extract_series(node, "Daily steps")
    sed_series   = _extract_series(node, "Sedentary (minutes)")
    act_series   = _extract_series(node, "Active minutes")
    slp_series   = _extract_series(node, "Sleep duration (hours)")

    avg_steps = _avg_before_day(steps_series, end_day)
    avg_sed   = _avg_before_day(sed_series, end_day)
    avg_act   = _avg_before_day(act_series, end_day)
    avg_slp   = _avg_before_day(slp_series, end_day)

    payload = {
        "Average daily steps": round(avg_steps, 2) if avg_steps is not None else "missing",
        "Average sedentary time (mins)": round(avg_sed, 2) if avg_sed is not None else "missing",
        "Average active time (mins)": round(avg_act, 2) if avg_act is not None else "missing",
        "Average sleep duration (hours)": round(avg_slp, 2) if avg_slp is not None else "missing",
    }

    block_json = json.dumps(payload, ensure_ascii=False, indent=2)

    lines = []
    lines.append("Below is a record of your historical activity and sleep summaries:")
    lines.append(block_json)

    text = "\n".join(lines)
    return text if text.endswith("\n") else text + "\n"

# --------------------
# Instructions block
# --------------------
def build_instructions_block(pid: str, target_day: int) -> str:
    instructions = (
f"""- **Question:** It has been {target_day} days since your transplant. Will you complete today's mood survey?
- **Explain:** Give one reason to complete the survey and one reason not to; then state your final decision for today (Yes / No).
- **Estimate:** Report the probability that your decision is correct as a single number between 0 and 1 (e.g., 0.73).

At the end, output a single JSON object with exactly these keys (no extra text):
{{"id": "{pid}", "day": {target_day}, "reason_do": "<string>", "reason_not": "<string>", "decision": "<Yes|No>", "confidence": <float between 0 and 1>}}
"""
    )
    return instructions

# --------------------
# Partner finder
# --------------------
def find_partner_pid(pid: str, demo: Dict[str, Any]) -> Optional[str]:
    dyad_id = demo.get("dyad_id")
    if dyad_id is None:
        return None
    pids = dyad_to_pids.get(str(dyad_id), [])
    if not pids:
        return None

    own_role = demo.get("role")
    want = "Patients" if own_role == "Caregivers" else ("Caregivers" if own_role == "Patients" else None)
    if want:
        for other in pids:
            if other == pid:
                continue
            other_demo = participants.get(other, {}).get("Demographic", {}) or {}
            if other_demo.get("role") == want:
                return other
    for other in pids:
        if other != pid:
            return other
    return None

# --------------------
# Sampling
# --------------------
random.seed(RANDOM_SEED)
sampled_patients   = random.sample(patient_ids,   k=min(N_PATIENTS, len(patient_ids)))
sampled_caregivers = random.sample(caregiver_ids, k=min(N_CAREGIVERS, len(caregiver_ids)))

print(f"[INFO] Sampled patients: {len(sampled_patients)}")
print(f"[INFO] Sampled caregivers: {len(sampled_caregivers)}")

# --------------------
# Ensure output directory
# --------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------
# Build prompts
# --------------------
def build_prompts_for_group(pids: List[str], target_day: int) -> Dict[str, str]:
    out: Dict[str, str] = {}

    for pid in pids:
        node = participants.get(pid, {}) or {}
        demo = node.get("Demographic", {}) or {}

        # intro
        intro = build_role_intro(demo, include_decision_lead_in=(target_day != 0))

        # own bg
        own_bg = build_own_background_block(demo)

        # partner bg
        partner_pid = find_partner_pid(pid, demo)
        if partner_pid:
            partner_demo = participants.get(partner_pid, {}).get("Demographic", {}) or {}
            partner_bg = build_partner_background_block(demo, partner_demo) if partner_demo else ""
        else:
            partner_bg = ""

        # assemble
        prompt_parts: List[str] = [intro, own_bg]
        if partner_bg:
            prompt_parts.append(partner_bg)

        if target_day != 0:
            # Mood last-7-day block
            prompt_parts.append(build_mood_context_block_last7(node, end_day=target_day))
            # NEW: Mood historical summary (prior to today)
            prompt_parts.append(build_mood_summary_stats_block(node, end_day=target_day))
            # Activity/Sleep averages up to target_day-1
            prompt_parts.append(build_activity_sleep_avgs_block(node, end_day=target_day))

        # Instructions
        prompt_parts.append(build_instructions_block(pid, target_day))

        out[pid] = "\n".join(part for part in prompt_parts if part).strip() + "\n"

    return out

# --------------------
# Generate and save
# --------------------
for day in TARGET_DAYS:
    # Patients
    patient_prompts = build_prompts_for_group(sampled_patients, target_day=day)
    p_out = Path(OUTPUT_DIR) / f"patients_day{day}.json"
    with p_out.open("w", encoding="utf-8") as f:
        json.dump(patient_prompts, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote patient prompts for day {day} -> {p_out.resolve()}")

    # Caregivers
    caregiver_prompts = build_prompts_for_group(sampled_caregivers, target_day=day)
    c_out = Path(OUTPUT_DIR) / f"caregivers_day{day}.json"
    with c_out.open("w", encoding="utf-8") as f:
        json.dump(caregiver_prompts, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote caregiver prompts for day {day} -> {c_out.resolve()}")

print("[DONE] All prompts with mood summary and averages generated.")
