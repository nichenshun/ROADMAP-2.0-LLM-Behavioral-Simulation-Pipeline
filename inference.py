#!/usr/bin/env python3
"""
run_experiment.py

Usage (from Linux shell), e.g.:

    python run_experiment.py patients 20 \
        --api-key YOUR_API_KEY \
        --endpoint https://YOUR-RESOURCE.openai.azure.com/ \
        --deployment gpt-5-mini

    python run_experiment.py caregivers 40 \
        --api-key YOUR_API_KEY \
        --endpoint https://YOUR-RESOURCE.openai.azure.com/ \
        --deployment gpt-5-mini

Arguments:
    role       : "patients" or "caregivers"
    day        : integer day to predict (e.g., 0, 20, 40, 60, 80, 100, 120)
    --api-key  : Azure OpenAI API key (if omitted, will try AZURE_OPENAI_API_KEY env var)
    --endpoint : Azure OpenAI endpoint (if omitted, will try AZURE_OPENAI_ENDPOINT env var)
    --deployment : Azure OpenAI deployment name (if omitted, will try AZURE_OPENAI_DEPLOYMENT env var)

What it does:
  - Loads prompts/{role}_day{day}.json
  - Iterates over **all** participant prompts in that file
  - For each participant:
      * Calls Azure OpenAI with a fixed system prompt + that user prompt
      * Immediately saves/updates one record in predictions/{role}_day{day}_results.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

from openai import AzureOpenAI


SYSTEM_PROMPT = """You are role-playing a specific participant enrolled in the ROADMAP 2.0 study, a randomized clinical trial at the University of Michigan Blood and Marrow Transplant Program to evaluate a mobile health app intervention designed to improve caregiver quality of life during their partner's hematopoietic cell transplantation.

A total of 166 caregiver-patient dyads were followed for 120 days post-transplant. All dyads received Fitbit devices and access to the app for mood reporting and physiological monitoring, while caregivers in the intervention arm also received positive psychology messages. 

Your goal is to predict whether the participant will complete today's mood-survey based on the participant's demographic characteristics and their prior mood-survey completion history. 

A participant is asked to enter their mood on a 1-10 scale once each day in the ROADMAP 2.0 app. 

Focus on simulating responses that are realistic, empathetic, and context-aware given the information provided. 

Think and respond as the participant would, avoiding generic or moralizing language. 

Briefly reflect (1 sentence) on the most relevant factors that influence today's completion. 

Then state your prediction clearly as one of: “Yes” or “No”. 

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
            parsed = json.loads(content)
        except Exception:
            parsed = None

    return {
        "raw_output": content,
        "parsed_json": parsed,
    }


def load_existing_results(path: Path) -> Dict[str, Any]:
    """
    Load existing results dict from path, or return empty dict if file doesn't exist.
    """
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        else:
            print(f"[WARN] Existing results file {path} is not a dict; overwriting with new dict.")
            return {}
    except Exception as e:
        print(f"[WARN] Could not read existing results file {path}: {e}; starting fresh.")
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ROADMAP 2.0 experiment via Azure OpenAI for a given role and day (all participants in that JSON)."
    )
    parser.add_argument(
        "role",
        choices=["patients", "caregivers"],
        help="Participant group: 'patients' or 'caregivers'.",
    )
    parser.add_argument(
        "day",
        type=int,
        help="Day to predict (e.g., 0, 20, 40, 60, 80, 100, 120).",
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

    args = parser.parse_args()
    role: str = args.role
    day: int = args.day

    # Resolve credentials: CLI args take priority, then env vars
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
        print("ERROR: Missing required Azure OpenAI configuration:\n  - " + "\n  - ".join(missing), file=sys.stderr)
        sys.exit(1)

    prompts_dir = Path("prompts3")
    predictions_dir = Path("predictions3")
    predictions_dir.mkdir(parents=True, exist_ok=True)

    prompts_file = prompts_dir / f"{role}_day{day}.json"
    if not prompts_file.exists():
        print(f"ERROR: prompts file not found: {prompts_file.resolve()}", file=sys.stderr)
        sys.exit(1)

    # Load prompts JSON: {participant_id: prompt_string, ...}
    with prompts_file.open("r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    if not isinstance(prompts_data, dict) or not prompts_data:
        print(f"ERROR: prompts JSON is empty or not a dict: {prompts_file}", file=sys.stderr)
        sys.exit(1)

    participant_ids = list(prompts_data.keys())
    print(f"[INFO] Running experiment for {role} on day {day}")
    print(f"[INFO] Using prompts file: {prompts_file.name}")
    print(f"[INFO] Total participants in this file: {len(participant_ids)}")

    # Init client
    client = create_client(api_key=api_key, endpoint=endpoint)

    # Results file (aggregate over all participants for this role+day)
    out_file = predictions_dir / f"{role}_day{day}_results.json"

    # Load existing results if any, so we can resume / overwrite per participant
    results_dict = load_existing_results(out_file)

    # Iterate over all participants
    for idx, pid in enumerate(participant_ids, start=1):
        user_prompt = prompts_data[pid]

        print(f"[INFO] ({idx}/{len(participant_ids)}) Running inference for participant ID: {pid}")

        # Call model
        result = call_model(client, deployment, user_prompt)

        # Build record
        rec = {
            "role": role,
            "day": day,
            "prompts_file": prompts_file.name,
            "participant_id": pid,
            "user_prompt": user_prompt,
            "model_response_raw": result["raw_output"],
            "model_response_parsed": result["parsed_json"],
        }

        # Update in-memory results dict for this participant
        results_dict[pid] = rec

        # Immediately write to disk after this participant
        try:
            with out_file.open("w", encoding="utf-8") as f:
                json.dump(results_dict, f, ensure_ascii=False, indent=2)
            print(f"[OK] Saved/updated result for participant {pid} to {out_file.name}")
        except Exception as e:
            print(f"[ERROR] Failed to write result for participant {pid}: {e}", file=sys.stderr)
            # continue to next participant (or you could break if you want)

    print(f"[DONE] Finished all participants for {role} on day {day}.")
    print(f"[INFO] Final results file: {out_file.resolve()}")


if __name__ == "__main__":
    main()
