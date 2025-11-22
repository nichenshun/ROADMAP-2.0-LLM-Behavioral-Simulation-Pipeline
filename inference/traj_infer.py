#!/usr/bin/env python3
"""
run_trajectory_inference.py
Usage (from Linux shell), e.g.:
python run_trajectory_inference.py 20 \
--api-key YOUR_API_KEY \
--endpoint https://YOUR-RESOURCE.openai.azure.com/ \
--deployment gpt-5-mini

Arguments:
start_day : integer day to start prediction from (e.g., 20)
--api-key : Azure OpenAI API key (if omitted, will try AZURE_OPENAI_API_KEY env var)
--endpoint : Azure OpenAI endpoint (if omitted, will try AZURE_OPENAI_ENDPOINT env var)
--deployment : Azure OpenAI deployment name (if omitted, will try AZURE_OPENAI_DEPLOYMENT env var)
--time-step : how many days to predict at once (default: 1)
--participant-id : specific participant ID to run (default: run all participants)
--participant-range : range of participant indices to run (e.g., "0-10") - relative to all participants
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from openai import AzureOpenAI

SYSTEM_PROMPT = """You are role-playing a specific participant enrolled in the ROADMAP 2.0 study, a randomized clinical trial at the University of Michigan Blood and Marrow Transplant Program to evaluate a mobile health app intervention designed to improve caregiver quality of life during their partner's hematopoietic cell transplantation.

A total of 166 caregiver-patient dyads were followed for 120 days post-transplant. All dyads received Fitbit devices and access to the app for mood reporting and physiological monitoring, while caregivers in the intervention arm also received positive psychology messages.

Your goal is to predict whether the participant will complete today's mood-survey based on the participant's demographic characteristics and their prior mood-survey completion history.

A participant is asked to enter their mood on a 1-10 scale once each day in the ROADMAP 2.0 app.

Focus on simulating responses that are realistic, empathetic, and context-aware given the information provided.

Think and respond as the participant would, avoiding generic or moralizing language.

Briefly reflect (1 sentence) on the most relevant factors that influence today's completion.

Then state your prediction clearly as one of: "Yes" or "No".

Keep your output concise and grounded in the provided context.
"""

def create_client(api_key: str, endpoint: str) -> AzureOpenAI:
    """Create Azure OpenAI client with given credentials."""
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2025-02-01-preview",  # update if needed
        azure_endpoint=endpoint,
    )
    return client

def call_model(client: AzureOpenAI, deployment: str, user_prompt: str) -> Dict[str, Any]:
    """
    Call Azure OpenAI chat completions with fixed system prompt and given user prompt.
    Returns dict with raw and parsed JSON output.
    """
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        # temperature=0.7,
    )
    
    content = ""
    if response.choices:
        msg = response.choices[0].message
        if msg and msg.content:
            content = msg.content.strip()
    
    parsed: Optional[Dict[str, Any]] = None
    if content:
        try:
            # Extract JSON from response if it's embedded in text
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = content[start_idx:end_idx]
                parsed = json.loads(json_str)
            else:
                parsed = None
        except Exception as e:
            print(f"[WARN] Could not parse JSON from response: {e}")
            parsed = None
    
    return {
        "raw_output": content,
        "parsed_json": parsed,
    }

def load_participants_data() -> Dict[str, Any]:
    """Load participants data from JSON file."""
    data_file = Path("participants_data.json")
    if not data_file.exists():
        print(f"ERROR: participants_data.json not found at {data_file.resolve()}", file=sys.stderr)
        sys.exit(1)
    
    with data_file.open("r", encoding="utf-8") as f:
        return json.load(f)

def get_participant_role(demo: Dict[str, Any]) -> str:
    """Determine participant role from demographic data."""
    if demo["role"] == "Patients":
        return "patient"
    elif demo["role"] == "Caregivers":
        return "caregiver"
    else:
        return "unknown"

def calculate_mood_stats(mood_data: Dict[str, Any], current_day: int) -> Tuple[int, str]:
    """Calculate missing days and average mood up to current day."""
    missing_count = 0
    mood_sum = 0.0
    mood_count = 0
    
    for day in range(1, current_day):
        day_key = f"Day {day}"
        if day_key in mood_data:
            mood_value = mood_data[day_key]
            if mood_value != "missing" and mood_value is not None:
                try:
                    mood_sum += float(mood_value)
                    mood_count += 1
                except (ValueError, TypeError):
                    missing_count += 1
            else:
                missing_count += 1
        else:
            missing_count += 1
    
    if mood_count > 0:
        avg_mood = f"{mood_sum / mood_count:.1f}"
    else:
        avg_mood = "NA"
    
    return missing_count, avg_mood

def get_complete_mood_history(mood_data: Dict[str, Any], current_day: int) -> Dict[str, Any]:
    """Get complete mood history from Day 0 to current_day-1."""
    complete_history = {}
    for day in range(0, current_day):
        day_key = f"Day {day}"
        if day_key in mood_data:
            complete_history[day_key] = mood_data[day_key]
        else:
            complete_history[day_key] = "missing"
    return complete_history

def get_recent_mood_history(mood_data: Dict[str, Any], current_day: int, window: int = 7) -> Dict[str, Any]:
    """Get mood history for the recent window days."""
    recent_mood = {}
    for i in range(window, 0, -1):
        day = current_day - i
        if day >= 0:  # Include day 0 and after
            day_key = f"Day {day}"
            if day_key in mood_data:
                recent_mood[day_key] = mood_data[day_key]
            else:
                recent_mood[day_key] = "missing"
    return recent_mood

def calculate_activity_stats(activity_data: Dict[str, Any], current_day: int) -> Dict[str, Any]:
    """Calculate average activity metrics up to current day."""
    metrics = {
        "steps": [],
        "sedentary": [],
        "active": [],
        "sleep": []
    }
    
    for day in range(0, current_day):  # Include day 0
        day_key = f"Day {day}"
        if day_key in activity_data.get("Daily steps", {}):
            metrics["steps"].append(activity_data["Daily steps"][day_key])
        if day_key in activity_data.get("Sedentary (minutes)", {}):
            metrics["sedentary"].append(activity_data["Sedentary (minutes)"][day_key])
        if day_key in activity_data.get("Active minutes", {}):
            metrics["active"].append(activity_data["Active minutes"][day_key])
        if day_key in activity_data.get("Sleep duration (hours)", {}):
            metrics["sleep"].append(activity_data["Sleep duration (hours)"][day_key])
    
    result = {}
    if metrics["steps"]:
        result["Average daily steps"] = round(sum(metrics["steps"]) / len(metrics["steps"]), 2)
    else:
        result["Average daily steps"] = "NA"
    
    if metrics["sedentary"]:
        result["Average sedentary time (mins)"] = round(sum(metrics["sedentary"]) / len(metrics["sedentary"]), 2)
    else:
        result["Average sedentary time (mins)"] = "NA"
    
    if metrics["active"]:
        result["Average active time (mins)"] = round(sum(metrics["active"]) / len(metrics["active"]), 2)
    else:
        result["Average active time (mins)"] = "NA"
    
    if metrics["sleep"]:
        result["Average sleep duration (hours)"] = round(sum(metrics["sleep"]) / len(metrics["sleep"]), 2)
    else:
        result["Average sleep duration (hours)"] = "NA"
    
    return result

def create_patient_prompt(participant_data: Dict[str, Any], participant_id: str, current_day: int) -> str:
    """Create patient-specific user prompt."""
    demo = participant_data["Demographic"]
    mood_data = participant_data.get("Mood", {})
    
    background = f"""- **Your Background:**
- You are in the {demo['age']} age group.
- Your gender is {demo['gender']}.
- Your monthly income is {demo['monthly_income']}.
- You are assigned to the {demo['arm']} arm of the study.
- Your transplant type is {demo['transplant_type']}.
- You stayed in the hospital for {demo['in_hospital_days']}.
- You receive caregiving from your caregiver for {demo['cg_hours']} per week.

- **Your caregiver partner's background information:**
- Your caregiver is in the {demo.get('cg_age', '61+')} age group.
- Your caregiver's gender is {demo.get('cg_gender', 'Female')}."""
    
    # For day 0, use simplified prompt
    if current_day == 0:
        return f"""You are a patient participant in a dyad in the ROADMAP 2.0 mobile-health study. 

You are a patient recovering from a hematopoietic cell transplantation (HCT) to treat a blood-related disease such as leukemia or lymphoma. During recovery, you may experience fatigue, nausea, pain, sleep disturbances, and emotional stress, while relying on your care partner and medical team to help you regain strength and prevent complications like infections or graft-versus-host disease. 

Each day, you are asked to enter your mood on a 1-10 scale (1=worst, 10=best). 

The study seeks to understand links between sensor data and well-being and to support caregivers during and after transplant. 

Each time step in this simulation represents one day.

Below is your background and history with the program. Based on this information, decide whether you will complete today's mood survey.

{background}

- **Question:** It has been {current_day} days since your transplant. Will you complete today's mood survey?
- **Explain:** Give one reason to complete the survey and one reason not to; then state your final decision for today (Yes / No).
- **Estimate:** Report the probability that your decision is correct as a single number between 0 and 1 (e.g., 0.73).

At the end, output a single JSON object with exactly these keys (no extra text):
{{"id": "{participant_id}", "day": {current_day}, "reason_do": "<string>", "reason_not": "<string>", "decision": "<Yes|No>", "confidence": <float between 0 and 1>, "mood_score": <if decision is Yes, provide a number between 1-10; if No, use "missing">}}
"""
    
    # For days >= 1, include complete history and statistics
    missing_days, avg_mood = calculate_mood_stats(mood_data, current_day)
    complete_history = get_complete_mood_history(mood_data, current_day)
    activity_stats = calculate_activity_stats(participant_data, current_day)
    
    return f"""You are a patient participant in a dyad in the ROADMAP 2.0 mobile-health study. 

You are a patient recovering from a hematopoietic cell transplantation (HCT) to treat a blood-related disease such as leukemia or lymphoma. During recovery, you may experience fatigue, nausea, pain, sleep disturbances, and emotional stress, while relying on your care partner and medical team to help you regain strength and prevent complications like infections or graft-versus-host disease. 

Each day, you are asked to enter your mood on a 1-10 scale (1=worst, 10=best). 

The study seeks to understand links between sensor data and well-being and to support caregivers during and after transplant. 

Each time step in this simulation represents one day.

Below is your background and history with the program. Based on this information, decide whether you will complete today's mood survey.

{background}

- **Context: past behavior:**
Below is a record of your previous mood survey record (each representing one day):
{json.dumps(complete_history, indent=2)}

Below is a summary of your historical mood completion prior to today:
{{
  "Missing days prior to Day {current_day}": {missing_days},
  "Average mood on completed days prior to Day {current_day}": {avg_mood}
}}

Below is a record of your historical activity and sleep summaries:
{json.dumps(activity_stats, indent=2)}

- **Question:** Today is Day {current_day}. Will you complete today's mood survey?
- **Explain:** Give one reason to complete the survey and one reason not to; then state your final decision for today (Yes / No).
- **Estimate:** Report the probability that your decision is correct as a single number between 0 and 1 (e.g., 0.73).

At the end, output a single JSON object with exactly these keys (no extra text):
{{"id": "{participant_id}", "day": {current_day}, "reason_do": "<string>", "reason_not": "<string>", "decision": "<Yes|No>", "confidence": <float between 0 and 1>, "mood_score": <if decision is Yes, provide a number between 1-10; if No, use "missing">}}
"""

def create_caregiver_prompt(participant_data: Dict[str, Any], participant_id: str, current_day: int) -> str:
    """Create caregiver-specific user prompt."""
    demo = participant_data["Demographic"]
    mood_data = participant_data.get("Mood", {})
    
    background = f"""- **Your Background:**
- You are in the {demo['age']} age group.
- Your gender is {demo['gender']}.
- Your monthly income is {demo['monthly_income']}.
- You provide caregiving to your patient for {demo['cg_hours']} per week.

- **Your patient partner's background information:**
- Your patient is in the {demo.get('cg_age', '61+')} age group.
- Your patient's gender is {demo.get('cg_gender', 'Female')}.
- Your patient's monthly income is {demo.get('cg_income', '$3,000 - $4,999')}.
- Your patient is assigned to the {demo['arm']} arm of the study.
- Your patient's transplant type is {demo['transplant_type']}.
- Your patient stayed in the hospital for {demo['in_hospital_days']}."""
    
    # For day 0, use simplified prompt
    if current_day == 0:
        return f"""You are a caregiver participant in a dyad in the ROADMAP 2.0 mobile-health study. 

You are a caregiver for a loved one recovering from a hematopoietic cell transplantation (HCT) for a serious blood or immune disorder. You provide daily physical and emotional support, monitor for complications, manage medications and appointments, and balance these duties with your own daily work routine. 

Each day, you are asked to enter your mood on a 1-10 scale (1=worst, 10=best). 

The study seeks to understand links between sensor data and well-being and to support caregivers during and after transplant. 

Each time step in this simulation represents one day.

Below is your background and history with the program. Based on this information, decide whether you will complete today's mood survey.

{background}

- **Question:** It has been {current_day} days since your transplant. Will you complete today's mood survey?
- **Explain:** Give one reason to complete the survey and one reason not to; then state your final decision for today (Yes / No).
- **Estimate:** Report the probability that your decision is correct as a single number between 0 and 1 (e.g., 0.73).

At the end, output a single JSON object with exactly these keys (no extra text):
{{"id": "{participant_id}", "day": {current_day}, "reason_do": "<string>", "reason_not": "<string>", "decision": "<Yes|No>", "confidence": <float between 0 and 1>, "mood_score": <if decision is Yes, provide a number between 1-10; if No, use "missing">}}
"""
    
    # For days >= 1, include recent history and statistics
    missing_days, avg_mood = calculate_mood_stats(mood_data, current_day)
    recent_mood = get_recent_mood_history(mood_data, current_day)
    activity_stats = calculate_activity_stats(participant_data, current_day)
    
    return f"""You are a caregiver participant in a dyad in the ROADMAP 2.0 mobile-health study. 

You are a caregiver for a loved one recovering from a hematopoietic cell transplantation (HCT) for a serious blood or immune disorder. You provide daily physical and emotional support, monitor for complications, manage medications and appointments, and balance these duties with your own daily work routine. 

Each day, you are asked to enter your mood on a 1-10 scale (1=worst, 10=best). 

The study seeks to understand links between sensor data and well-being and to support caregivers during and after transplant. 

Each time step in this simulation represents one day.

Below is your background and history with the program. Based on this information, decide whether you will complete today's mood survey.

{background}

- **Context: past behavior:**
Below is a record of your mood surveys from the previous 7 days (each entry represents one day):
{json.dumps(recent_mood, indent=2)}

Below is a summary of your historical mood completion prior to today:
{{
  "Missing days prior to Day {current_day}": {missing_days},
  "Average mood on completed days prior to Day {current_day}": {avg_mood}
}}

Below is a record of your historical activity and sleep summaries:
{json.dumps(activity_stats, indent=2)}

- **Question:** It has been {current_day} days since your transplant. Will you complete today's mood survey?
- **Explain:** Give one reason to complete the survey and one reason not to; then state your final decision for today (Yes / No).
- **Estimate:** Report the probability that your decision is correct as a single number between 0 and 1 (e.g., 0.73).

At the end, output a single JSON object with exactly these keys (no extra text):
{{"id": "{participant_id}", "day": {current_day}, "reason_do": "<string>", "reason_not": "<string>", "decision": "<Yes|No>", "confidence": <float between 0 and 1>, "mood_score": <if decision is Yes, provide a number between 1-10; if No, use "missing">}}
"""

def create_user_prompt(participant_data: Dict[str, Any], participant_id: str, current_day: int) -> str:
    """Create user prompt for a specific participant and day based on role."""
    demo = participant_data["Demographic"]
    role = get_participant_role(demo)
    
    if role == "patient":
        return create_patient_prompt(participant_data, participant_id, current_day)
    elif role == "caregiver":
        return create_caregiver_prompt(participant_data, participant_id, current_day)
    else:
        print(f"ERROR: Unknown role for participant {participant_id}: {demo['role']}", file=sys.stderr)
        sys.exit(1)

def update_mood_data(mood_data: Dict[str, Any], prediction: Dict[str, Any], current_day: int) -> None:
    """Update mood data with prediction result."""
    day_key = f"Day {current_day}"
    if prediction.get("decision") == "Yes":
        # Use the mood_score from the prediction if available
        if "mood_score" in prediction and prediction["mood_score"] != "missing":
            try:
                mood_score = float(prediction["mood_score"])
                mood_data[day_key] = mood_score
            except (ValueError, TypeError):
                # Fallback to generating a mood score if prediction doesn't have valid mood_score
                recent_moods = []
                for day in range(max(0, current_day-7), current_day):  # Include day 0
                    prev_key = f"Day {day}"
                    if prev_key in mood_data and mood_data[prev_key] != "missing":
                        try:
                            recent_moods.append(float(mood_data[prev_key]))
                        except (ValueError, TypeError):
                            continue
                
                if recent_moods:
                    # Use average of recent moods with small random variation
                    import random
                    avg_mood = sum(recent_moods) / len(recent_moods)
                    mood_score = max(1, min(10, round(avg_mood + random.uniform(-0.5, 0.5), 1)))
                else:
                    # Default to moderate mood if no history
                    mood_score = 6.5
                mood_data[day_key] = mood_score
        else:
            # Generate mood score if not provided in prediction
            recent_moods = []
            for day in range(max(0, current_day-7), current_day):  # Include day 0
                prev_key = f"Day {day}"
                if prev_key in mood_data and mood_data[prev_key] != "missing":
                    try:
                        recent_moods.append(float(mood_data[prev_key]))
                    except (ValueError, TypeError):
                        continue
            
            if recent_moods:
                # Use average of recent moods with small random variation
                import random
                avg_mood = sum(recent_moods) / len(recent_moods)
                mood_score = max(1, min(10, round(avg_mood + random.uniform(-0.5, 0.5), 1)))
            else:
                # Default to moderate mood if no history
                mood_score = 6.5
            mood_data[day_key] = mood_score
    else:
        mood_data[day_key] = "missing"

def save_prompt(trajectory_dir: Path, participant_id: str, day: int, prompt: str) -> None:
    """Save prompt to file."""
    prompt_file = trajectory_dir / participant_id / f"prompt_{day}.txt"
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    
    with prompt_file.open("w", encoding="utf-8") as f:
        f.write(prompt)

def save_prediction(trajectory_dir: Path, participant_id: str, predictions: List[Dict[str, Any]]) -> None:
    """Save all predictions for a participant to JSON file."""
    prediction_file = trajectory_dir / participant_id / "predictions.json"
    prediction_file.parent.mkdir(parents=True, exist_ok=True)
    
    with prediction_file.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

def parse_participant_range(range_str: str, total_participants: int) -> List[int]:
    """Parse participant range string like '0-10' and return list of indices."""
    try:
        if '-' in range_str:
            start, end = map(int, range_str.split('-'))
            if start < 0 or end >= total_participants or start > end:
                raise ValueError(f"Range {range_str} is invalid for {total_participants} participants")
            return list(range(start, end + 1))
        else:
            # Single number
            idx = int(range_str)
            if idx < 0 or idx >= total_participants:
                raise ValueError(f"Index {idx} is out of range for {total_participants} participants")
            return [idx]
    except ValueError as e:
        print(f"ERROR: Invalid participant range format: {e}", file=sys.stderr)
        sys.exit(1)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ROADMAP 2.0 trajectory inference from start day to day 120."
    )
    parser.add_argument(
        "--start_day",
        type=int,
        help="Day to start prediction from (e.g., 20).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Azure OpenAI API key (if omitted, will use AZURE_OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="Azure OpenAI endpoint (if omitted, will use AZURE_OPENAI_ENDPOINT env var).",
    )
    parser.add_argument(
        "--deployment",
        type=str,
        default=None,
        help="Azure OpenAI deployment name (if omitted, will use AZURE_OPENAI_DEPLOYMENT env var).",
    )
    parser.add_argument(
        "--time-step",
        type=int,
        default=1,
        help="How many days to predict at once (default: 1).",
    )
    parser.add_argument(
        "--participant-id",
        type=str,
        default=None,
        help="Specific participant ID to run (default: run all participants).",
    )
    parser.add_argument(
        "--participant-range",
        type=str,
        default=None,
        help="Range of participant indices to run (e.g., '0-10') - relative to all participants.",
    )
    
    args = parser.parse_args()
    start_day: int = args.start_day
    time_step: int = args.time_step
    
    # Resolve credentials
    api_key = args.api_key or os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = args.endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
    deployment = args.deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    
    missing = []
    if not api_key:
        missing.append("API key (either --api-key or AZURE_OPENAI_API_KEY)")
    if not endpoint:
        missing.append("endpoint (either --endpoint or AZURE_OPENAI_ENDPOINT)")
    if not deployment:
        missing.append("deployment (either --deployment or AZURE_OPENAI_DEPLOYMENT)")
    if missing:
        print("ERROR: Missing required Azure OpenAI configuration:\n - " + "\n - ".join(missing), file=sys.stderr)
        sys.exit(1)
    
    # Load participants data
    participants_data = load_participants_data()
    
    # Get ordered list of all participant IDs
    ordered_ids = list(participants_data.keys())
    total_participants = len(ordered_ids)
    
    if not participants_data:
        print(f"ERROR: No participants found in data file", file=sys.stderr)
        sys.exit(1)
    
    print(f"[INFO] Found {total_participants} total participants")
    
    # Apply participant range filter if specified
    if args.participant_range:
        indices = parse_participant_range(args.participant_range, total_participants)
        target_participants = {}
        for idx in indices:
            if idx < len(ordered_ids):
                pid = ordered_ids[idx]
                target_participants[pid] = participants_data[pid]
        print(f"[INFO] Selected {len(target_participants)} participants by range {args.participant_range}: {list(target_participants.keys())}")
    elif args.participant_id:
        # Filter by specific participant ID
        if args.participant_id in participants_data:
            target_participants = {args.participant_id: participants_data[args.participant_id]}
        else:
            print(f"ERROR: Participant ID '{args.participant_id}' not found", file=sys.stderr)
            sys.exit(1)
    else:
        # Use all participants
        target_participants = participants_data
    
    if not target_participants:
        print(f"ERROR: No participants selected for processing", file=sys.stderr)
        sys.exit(1)
    
    # Count roles for logging
    role_counts = {"patient": 0, "caregiver": 0}
    for participant_data in target_participants.values():
        role = get_participant_role(participant_data["Demographic"])
        if role in role_counts:
            role_counts[role] += 1
    
    print(f"[INFO] Running trajectory inference from day {start_day} to 120")
    print(f"[INFO] Time step: {time_step} day(s)")
    print(f"[INFO] Participants to process: {len(target_participants)}")
    print(f"[INFO] Role distribution: {role_counts['patient']} patients, {role_counts['caregiver']} caregivers")
    
    # Create trajectory directory
    trajectory_dir = Path(f"traj_{start_day}")
    trajectory_dir.mkdir(exist_ok=True)
    
    # Init client
    client = create_client(api_key=api_key, endpoint=endpoint)
    
    # Process each participant
    for participant_id, participant_data in target_participants.items():
        role = get_participant_role(participant_data["Demographic"])
        print(f"[INFO] Processing {role} participant: {participant_id}")
        
        all_predictions = []
        current_mood_data = participant_data.get("Mood", {}).copy()
        
        # Run inference from start_day to 120
        for current_day in range(start_day, 121, time_step):
            print(f"[INFO] Predicting day {current_day} for {participant_id}")
            
            # Create prompt for current day
            user_prompt = create_user_prompt(participant_data, participant_id, current_day)
            
            # Save prompt
            save_prompt(trajectory_dir, participant_id, current_day, user_prompt)
            
            # Call model
            result = call_model(client, deployment, user_prompt)
            
            # Build prediction record
            prediction_record = {
                "day": current_day,
                "user_prompt": user_prompt,
                "model_response_raw": result["raw_output"],
                "model_response_parsed": result["parsed_json"],
            }
            all_predictions.append(prediction_record)
            
            # Update mood data with prediction for next iteration
            if result["parsed_json"]:
                update_mood_data(current_mood_data, result["parsed_json"], current_day)
                # Also update the participant_data for next day's calculations
                participant_data["Mood"] = current_mood_data
            
            print(f"[OK] Completed prediction for day {current_day}")
        
        # Save all predictions for this participant
        save_prediction(trajectory_dir, participant_id, all_predictions)
        print(f"[OK] Saved all predictions for {participant_id}")
    
    print(f"[DONE] Finished trajectory inference for all participants")
    print(f"[INFO] Results saved in: {trajectory_dir.resolve()}")

if __name__ == "__main__":
    main()