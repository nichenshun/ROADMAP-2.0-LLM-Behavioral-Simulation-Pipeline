# ROADMAP-2.0-LLM-Behavioral-Simulation-Pipeline

**Simulation and prediction pipeline for participant behavior in the ROADMAP 2.0 mobile-health study using large language models (LLMs).**

This repository implements an end-to-end experimental framework for simulating and predicting participant mood-survey completion behavior in the *ROADMAP 2.0* clinical study.  
The project integrates data preprocessing, prompt generation, Azure OpenAI inference, and evaluation modules—enabling reproducible behavioral modeling and analysis of both caregivers and patients.

---

## 📂 Project Overview

### 🔹 Modules
| Script | Description |
|--------|--------------|
| ```data_process.py``` | Merges some of the study ```.csv```'s under ```/data/``` into a unified ```participants_data.json```, formatting behavioral and demographic records. |
| ```prompts_gen.py``` | Randomly samples **100 patients** and **100 caregivers**, then generates role-specific prompts for prediction days (0, 20, 40, 60, 80, 100, 120). |
| ```inference.py``` | Calls **Azure OpenAI API** to simulate participant decisions (Yes/No) for mood-survey completion, saving predictions incrementally to ```.json```. |
| ```eval.ipynb``` | Evaluates model predictions vs. ground truth (from ```participants_data.json```) and visualizes metrics such as TPR, FPR, overall accuracy, and cross-entropy loss. |

---
## 🧩 Pipeline Steps
### 1️⃣ Data Processing 
**Goal**: Build a unified `.json` dataset `participants_data.json` from multiple raw `.csv` files under `/data`.
**Input files**:
```bash
- demographic_data.csv
- daily_steps.csv
- daily_activity.csv
- sleep_stages.csv
- sleep_classic.csv
- mood.csv
```
**Run**:
```bash
python data_process.py
```
This produces a comprehensive file: 
``` bash
participants_data.json
```
**Example**: The structure for each participant (e.g., ```"P311"```) looks roughly like:
```bash
"P311": {
  "Demographic": {
    "arm": "Intervention",
    "cg_hours": "Less than or equal to 40 hours",
    "age": "61+",
    "gender": "Female",
    "monthly_income": "$1,000 - $2,999",
    "transplant_type": "Autologous",
    "dyad_id": 1,
    "role": "Caregivers",
    "in_hospital_days": "OHSU patient"
  },
  "Daily steps": {
    "Day 4": 11287.0,
    "Day 5": 17519.0,
    ...
  },
  "Sedentary (minutes)": {
    "Day 4": 1159.0,
    "Day 5": 1040.0,
    ...
  },
  "Active minutes": {
    "Day 4": 278.0,
    "Day 5": 353.0,
    ...
  },
  "Sleep duration (hours)": {
    "Day 18": 8.07,
    "Day 22": 7.79,
    ...
  },
  "Sleep duration (classic, hours)": {
    "Day 42": 2.45
  },
  "Mood": {
    "Day 4": 9.0,
    "Day 5": 8.0,
    ...
  }
}
```
### 2️⃣ Prompt Generation 
**Goal**: Generate natural-language prompts for the LLM, one per participant per target day, encoding:
- Participant’s demographic background
- Dyad partner’s background (caregiver ↔ patient)
- Historical mood completion pattern up to (but not including) the target day
- A standardized **Question / Explain / Estimate** instruction asking the model to decide whether the participant will complete **the target’s** mood survey
  
**Sampling strategy**: From ```participants_data.json```, the script ```prompts_gen.py``` randomly samples
- 100 patients
- 100 caregivers

All subsequent prompts and experiments are restricted to this sampled subset.

**Run**:
```bash
python prompts_gen.py
```
This will create prompt ```.json``` files under ```/prompts```:
```bash
prompts/
├── caregivers_day0.json
├── caregivers_day20.json
├── caregivers_day40.json
...
├── caregivers_day120.json

├── patients_day0.json
├── patients_day20.json
├── patients_day40.json
...
├── patients_day120.json
```
**Example**:
- System prompt:
```bash
You are role-playing a specific participant enrolled in the ROADMAP 2.0 study, a randomized clinical trial at the University of Michigan Blood and Marrow Transplant Program to evaluate a mobile health app intervention designed to improve caregiver quality of life during their partner’s hematopoietic cell transplantation.

A total of 166 caregiver–patient dyads were followed for 120 days post-transplant. All dyads received Fitbit devices and access to the app for mood reporting and physiological monitoring, while caregivers in the intervention arm also received positive psychology messages.

Your goal is to predict whether the participant will complete today’s mood-survey based on the participant’s demographic characteristics and their prior mood-survey completion history.

A participant is asked to enter their mood on a 1–10 scale once each day in the ROADMAP 2.0 app.

Focus on simulating responses that are realistic, empathetic, and context-aware given the information provided.

Think and respond as the participant would, avoiding generic or moralizing language.

Briefly reflect (1 sentence) on the most relevant factors that influence today’s completion.

Then state your prediction clearly as one of: “Yes” or “No”.

Keep your output concise and grounded in the provided context. 
```

### 3️⃣ Inference via Azure OpenAI
Run behavioral simulations using ```Azure OpenAI API```:
```bash
python inference.py
```
The script iterates through all participant prompts for the specified day and role,
calls the model (e.g., gpt-5-mini), and saves each result immediately to ```/predictions/```.




