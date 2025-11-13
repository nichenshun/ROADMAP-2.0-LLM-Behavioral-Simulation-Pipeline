#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate prompts for ROADMAP 2.0 LLM simulation with customized background wording.

NEW change in this version:
- If target day == 0, DO NOT include the **Context: past behavior** block at all.
- FIXED: Mood history block now correctly reads keys of form "Day N" or "N",
         and fills only Day 0..Day (target_day-1) with numeric values or "missing" (lowercase).

Other previously requested rules remain:

- For CAREGIVERS, in **Your Background:**
  • REMOVE these items:
      - "You are assigned to the ... arm of the study."
      - "Your role is ...."
      - Any line about the PATIENT's transplant type.
      - Any line about the PATIENT's hospital stay.
  • CHANGE caregiving hours line to:
      - "You provide caregiving to your patient for <X> per week."

- In the CAREGIVER'S partner block (patient partner):
  • Title stays: "- **Your patient partner’s background information**"
  • Transform bullets (examples):
      - "You are in the 61+ age group."  -> "Your patient is in the 61+ age group."
      - "Your gender is Male."           -> "Your patient‘s gender is male."   (value lowercased)
      - "Your monthly income is Less than $1,000." -> "Your patient’s monthly income is less than $1,000." (value lowercased)
      - "You are assigned to the Intervention arm of the study." -> "Your patient is assigned to the Intervention arm of the study."
      - "Your role is Patients." -> REMOVE
      - "Your transplant type is Autologous." -> "Your patient’s transplant type is Autologous."
      - "You have stayed in the hospital for 15." -> "Your patient stayed in the hospital for 15 days."
        (Also handle 'OHSU patient' -> "Your patient is an OHSU patient.")

- For PATIENTS, in **Your Background:**
  • Keep: age, gender (lowercase value), monthly income, arm, transplant type.
  • REMOVE: role.
  • Hospital stay wording: "You stayed in the hospital for <N> days."
  • Caregiving hours wording: "You receive caregiving from your caregiver for <X> per week."

- In the PATIENT'S partner block (caregiver partner):
  • Title: "- **Your caregiver partner’s background information**"
  • Transform bullets (examples):
      - "You are in the 61+ age group."  -> "Your caregiver is in the 61+ age group."
      - "Your gender is Female."         -> "Your caregiver‘s gender is female." (value lowercased)
      - "Your monthly income is $5,000 - $6,999." -> "Your caregiver’s monthly income is $5,000 - $6,999."
  • REMOVE from partner block: arm, role, any 'caring for a patient ...', any 'patient ... hospital ...',
    any caregiving-hours bullet.

All other behavior (intro text by role, sampling 100 per role,
producing prompts/{role}_day{X}.json for days [0,20,40,60,80,100,120]) remains unchanged.

Run:
    python prompts_gen.py
"""

import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict
import os

# --------------------
# Config
# --------------------
DEMOGRAPHICS_JSON_PATH = "participants_data.json"  # keep as provided
OUTPUT_DIR = "prompts"

TARGET_DAYS = [0, 20, 40, 60, 80, 100, 120]   # include Day 0 as requested
N_PATIENTS = 100
N_CAREGIVERS = 100

MOOD_START_DAY = 0       # inclusive for history window
GLOBAL_MAX_DAY = 120     # safety upper bound for days in the JSON

RANDOM_SEED = 42         # reproducible sampling

# --------------------
# Load participants JSON
# --------------------
in_path = Path(DEMOGRAPHICS_JSON_PATH)
if not in_path.exists():
    raise FileNotFoundError(f"Input JSON not found at: {in_path.resolve()}")

with open(in_path, "r", encoding="utf-8") as f:
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

    role = demo.get("role")
    dyad_id = demo.get("dyad_id")
    tx = demo.get("transplant_type")
    hosp = demo.get("in_hospital_days")

    if role == "Patients":
        patient_ids.append(pid)
    elif role == "Caregivers":
        caregiver_ids.append(pid)

    if dyad_id is not None:
        key = str(dyad_id)
        dyad_to_pids[key].append(pid)

    # For caregiver description (patient info)
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

def _is_numeric_days(x: Any) -> bool:
    """Return True if x looks like an integer number of days (e.g., '15' or 15)."""
    try:
        int(str(x))
        return True
    except Exception:
        return False

def _parse_day_key(k: Any) -> Optional[int]:
    """
    Accept 'Day N' or 'N' -> int N; return None if not parsable or out of range.
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

# --------------------
# Role-specific introductory paragraph (with conditional last sentence for day 0)
# --------------------
def build_role_intro(demo: Dict[str, Any], include_decision_lead_in: bool) -> str:
    role = demo.get("role")
    if role == "Caregivers":
        role_word = "caregiver"
    elif role == "Patients":
        role_word = "patient"
    else:
        role_word = "participant"

    caregiver_intro_core = f"""You are a {role_word} participant in a dyad in the ROADMAP 2.0 mobile-health study. 

You are a caregiver for a loved one recovering from a hematopoietic cell transplantation (HCT) for a serious blood or immune disorder. You provide daily physical and emotional support, monitor for complications, manage medications and appointments, and balance these duties with your own daily work routine. 

Each day, you are asked to enter your mood on a 1-10 scale (1=worst, 10=best). 

The study seeks to understand links between sensor data and well-being and to support caregivers during and after transplant. 

Each time step in this simulation represents one day."""
    patient_intro_core = f"""You are a {role_word} participant in a dyad in the ROADMAP 2.0 mobile-health study. 

You are a patient recovering from a hematopoietic cell transplantation (HCT) to treat a blood-related disease such as leukemia or lymphoma. During recovery, you may experience fatigue, nausea, pain, sleep disturbances, and emotional stress, while relying on your care partner and medical team to help you regain strength and prevent complications like infections or graft-versus-host disease. 

Each day, you are asked to enter your mood on a 1-10 scale (1=worst, 10=best). 

The study seeks to understand links between sensor data and well-being and to support caregivers during and after transplant. 

Each time step in this simulation represents one day."""
    if role == "Caregivers":
        core = caregiver_intro_core
    else:
        core = patient_intro_core

    if include_decision_lead_in:
        core += """

Below is your background and history with the program. Based on this information, decide whether you will complete today's mood survey."""
    return core.strip() + "\n\n"

# --------------------
# OWN background bullets (with requested filtering/wording by role)
# --------------------
def build_own_background_bullets(demo: Dict[str, Any]) -> List[str]:
    """
    Build OWN background lines with the new rules.
    """
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
        gender_val = _to_lower(gender)  # lowercase value
        lines.append(f"- Your gender is {gender_val}.")
    if monthly_income:
        lines.append(f"- Your monthly income is {monthly_income}.")
    # Arm:
    if is_patient and arm:
        lines.append(f"- You are assigned to the {arm} arm of the study.")
    # Role: removed for both groups
    # Transplant type:
    if is_patient and transplant_type:
        lines.append(f"- Your transplant type is {transplant_type}.")
    # Hospital:
    if in_hospital_days:
        if str(in_hospital_days) == "OHSU patient":
            if is_patient:
                lines.append("- You are an OHSU patient.")
        else:
            if is_patient and _is_numeric_days(in_hospital_days):
                lines.append(f"- You stayed in the hospital for {int(in_hospital_days)} days.")
    # Caregiving hours:
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
    if not text.endswith("\n"):
        text += "\n"
    return text

# --------------------
# PARTNER background (transforms & filtering by perspective)
# --------------------
def build_partner_background_block(own_demo: Dict[str, Any], partner_demo: Dict[str, Any]) -> str:
    """
    Build partner background with requested transformations and removals.
    - If own role == Caregivers -> partner is patient: use "Your patient ..." phrasing.
    - If own role == Patients   -> partner is caregiver: use "Your caregiver ..." phrasing.
    Remove disallowed items per spec.
    """
    partner_role = partner_demo.get("role")

    if partner_role == "Caregivers":
        title = "- **Your caregiver partner's background information**"
        noun = "caregiver"
        keep_fields = {"age", "gender", "monthly_income"}  # remove arm/role/tx/hospital/cg_hours
        lower_gender = True
        lower_income = False
        include_arm = False
        include_tx = False
        include_hosp = False
    else:
        title = "- **Your patient partner's background information**"
        noun = "patient"
        keep_fields = {"age", "gender", "monthly_income", "arm", "transplant_type", "in_hospital_days"}
        lower_gender = True
        lower_income = True
        include_arm = True
        include_tx = True
        include_hosp = True

    age              = partner_demo.get("age")
    gender           = partner_demo.get("gender")
    monthly_income   = partner_demo.get("monthly_income")
    arm              = partner_demo.get("arm")
    transplant_type  = partner_demo.get("transplant_type")
    in_hospital_days = partner_demo.get("in_hospital_days")

    bullets: List[str] = [title]

    if "age" in keep_fields and age:
        bullets.append(f"- Your {noun} is in the {age} age group.")
    if "gender" in keep_fields and gender:
        g = _to_lower(gender) if lower_gender else gender
        bullets.append(f"- Your {noun}'s gender is {g}.")
    if "monthly_income" in keep_fields and monthly_income:
        inc = _to_lower(monthly_income) if lower_income else monthly_income
        bullets.append(f"- Your {noun}'s monthly income is {inc}.")
    if "arm" in keep_fields and include_arm and arm:
        bullets.append(f"- Your {noun} is assigned to the {arm} arm of the study.")
    if "transplant_type" in keep_fields and include_tx and transplant_type:
        bullets.append(f"- Your {noun}'s transplant type is {transplant_type}.")
    if "in_hospital_days" in keep_fields and include_hosp and in_hospital_days:
        if str(in_hospital_days) == "OHSU patient":
            bullets.append(f"- Your {noun} is an OHSU patient.")
        elif _is_numeric_days(in_hospital_days):
            bullets.append(f"- Your {noun} stayed in the hospital for {int(in_hospital_days)} days.")

    text = "\n".join(bullets)
    if not text.endswith("\n"):
        text += "\n"
    return text

# --------------------
# Mood context block up to (end_day-1)  — FIXED
# --------------------
def build_mood_context_block(node: Dict[str, Any], end_day: int) -> str:
    """
    History to display: Day 0 .. Day (end_day-1).
    - Accept Mood keys as "Day N" or "N".
    - If a day in this range has no entry -> "missing" (lowercase).
    - If 'end_day' <= 0, caller should skip adding this block entirely.
    """
    mood_map = node.get("Mood", {}) or {}

    # Normalize keys → int days
    day_to_mood: Dict[int, Any] = {}
    for k, v in mood_map.items():
        d = _parse_day_key(k)
        if d is None:
            continue
        try:
            val = float(v)
        except Exception:
            val = v
        day_to_mood[d] = val

    # Clamp (end_day-1) upper bound; ensure non-negative span.
    end_day_clamped = max(0, min(end_day, GLOBAL_MAX_DAY + 1))

    # Build dense dict Day 0 .. Day (end_day-1)
    mood_dict_for_prompt: Dict[str, Any] = {}
    for d in range(MOOD_START_DAY, end_day_clamped):
        key = f"Day {d}"
        if d in day_to_mood:
            mood_dict_for_prompt[key] = day_to_mood[d]
        else:
            mood_dict_for_prompt[key] = "missing"  # LOWERCASE per your example

    mood_json_str = json.dumps(mood_dict_for_prompt, ensure_ascii=False, indent=2)

    lines = []
    lines.append("- **Context: past behavior:**")
    lines.append("Below is a record of your previous mood survey record (each representing one day):")
    lines.append(mood_json_str)

    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    return text

# --------------------
# Instructions block with day & id and updated wording (Yes/No)
# --------------------
def build_instructions_block(pid: str, target_day: int) -> str:
    instructions = (
f"""- **Question:** Today is Day {target_day}. Will you complete today's mood survey?
- **Explain:** Give one reason to complete the survey and one reason not to; then state your final decision for today (Yes / No).
- **Estimate:** Report the probability that your decision is correct as a single number between 0 and 1 (e.g., 0.73).

At the end, output a single JSON object with exactly these keys (no extra text):
{{"id": "{pid}", "day": {target_day}, "reason_do": "<string>", "reason_not": "<string>", "decision": "<Yes|No>", "confidence": <float between 0 and 1>}}
"""
    )
    return instructions

# --------------------
# Find partner pid for a given pid
# --------------------
def find_partner_pid(pid: str, demo: Dict[str, Any]) -> Optional[str]:
    dyad_id = demo.get("dyad_id")
    if dyad_id is None:
        return None
    key = str(dyad_id)
    pids_in_dyad = dyad_to_pids.get(key, [])
    if not pids_in_dyad:
        return None

    own_role = demo.get("role")
    opposite_role = None
    if own_role == "Caregivers":
        opposite_role = "Patients"
    elif own_role == "Patients":
        opposite_role = "Caregivers"

    # prefer opposite role
    if opposite_role is not None:
        for other_pid in pids_in_dyad:
            if other_pid == pid:
                continue
            other_demo = participants.get(other_pid, {}).get("Demographic", {}) or {}
            if other_demo.get("role") == opposite_role:
                return other_pid

    # fallback: any other participant in same dyad
    for other_pid in pids_in_dyad:
        if other_pid != pid:
            return other_pid

    return None

# --------------------
# Sample 100 patients + 100 caregivers (or all if <100)
# --------------------
random.seed(RANDOM_SEED)

sampled_patients = random.sample(
    patient_ids, k=min(N_PATIENTS, len(patient_ids))
)
sampled_caregivers = random.sample(
    caregiver_ids, k=min(N_CAREGIVERS, len(caregiver_ids))
)

print(f"[INFO] Sampled patients: {len(sampled_patients)}")
print(f"[INFO] Sampled caregivers: {len(sampled_caregivers)}")

# --------------------
# Ensure output directory
# --------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------
# Build prompts for a set of participants and a target day (with all rules)
# --------------------
def build_prompts_for_group(pids: List[str], target_day: int) -> Dict[str, str]:
    out: Dict[str, str] = {}

    for pid in pids:
        node = participants.get(pid, {})
        if not isinstance(node, dict):
            continue

        demo = node.get("Demographic", {}) or {}
        if not isinstance(demo, dict) or len(demo) == 0:
            # minimal fallback
            prompt = (
                "You are a participant in the ROADMAP 2.0 mobile-health study.\n\n"
                "- **Your Background:**\n"
                + build_instructions_block(pid, target_day)
            )
            out[pid] = prompt
            continue

        # Intro: omit the last decision-lead-in sentence ONLY for day 0
        intro = build_role_intro(demo, include_decision_lead_in=(target_day != 0))

        # Own background
        own_bg_block = build_own_background_block(demo)

        # Partner background (if any)
        partner_pid = find_partner_pid(pid, demo)
        if partner_pid is not None:
            partner_demo = participants.get(partner_pid, {}).get("Demographic", {}) or {}
            partner_bg_block = build_partner_background_block(demo, partner_demo) if partner_demo else ""
        else:
            partner_bg_block = ""

        # Assemble base
        full_prompt = intro + own_bg_block
        if partner_bg_block:
            full_prompt += "\n" + partner_bg_block

        # Mood context up to (target_day - 1) — SKIP ENTIRELY when target_day == 0
        if target_day != 0:
            mood_block = build_mood_context_block(node, end_day=target_day)
            full_prompt += "\n" + mood_block

        # Instructions
        full_prompt += "\n" + build_instructions_block(pid, target_day)

        out[pid] = full_prompt

    return out

# --------------------
# Generate and save prompts for each experiment
# --------------------
for day in TARGET_DAYS:
    # Patients
    patient_prompts = build_prompts_for_group(sampled_patients, target_day=day)
    patient_out_path = Path(OUTPUT_DIR) / f"patients_day{day}.json"
    with open(patient_out_path, "w", encoding="utf-8") as f:
        json.dump(patient_prompts, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote patient prompts for day {day} to {patient_out_path.resolve()}")

    # Caregivers
    caregiver_prompts = build_prompts_for_group(sampled_caregivers, target_day=day)
    caregiver_out_path = Path(OUTPUT_DIR) / f"caregivers_day{day}.json"
    with open(caregiver_out_path, "w", encoding="utf-8") as f:
        json.dump(caregiver_prompts, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote caregiver prompts for day {day} to {caregiver_out_path.resolve()}")

print("[DONE] All prompt files updated with the requested changes.]")
