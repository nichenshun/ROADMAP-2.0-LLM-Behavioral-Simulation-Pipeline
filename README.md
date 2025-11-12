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
**Input files**
```bash
- demographic_data.csv
- daily_steps.csv
- daily_activity.csv
- sleep_stages.csv
- sleep_classic.csv
- mood.csv
```
**Run**
```bash
python data_process.py
```
This produces a comprehensive file: 
``` bash
participants_data.json
```
**Example**
The structure for each participant (e.g., ```"P311"```) looks roughly like:
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
**Goal**
Generate natural-language prompts for the LLM, one per participant per target day, encoding:
- Participant’s demographic background
- Dyad partner’s background (caregiver ↔ patient)
- Historical mood completion pattern up to (but not including) the target day
- A standardized **Question / Explain / Estimate** instruction asking the model to decide whether the participant will complete **the target’s** mood survey
  
**Sampling strategy**
From ```participants_data.json```, the script ```prompts_gen.py``` randomly samples
- 100 patients
- 100 caregivers

All subsequent prompts and experiments are restricted to this sampled subset.

**Run**
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
**Example**
- User prompt for patients (e.g., ```"P094"```, ```target=Day 20```):
```bash
You are a patient participant in a dyad in the ROADMAP 2.0 mobile-health study. 

You are a patient recovering from a hematopoietic cell transplantation (HCT) to treat a blood-related disease such as leukemia or lymphoma. During recovery, you may experience fatigue, nausea, pain, sleep disturbances, and emotional stress, while relying on your care partner and medical team to help you regain strength and prevent complications like infections or graft-versus-host disease. 

Each day, you are asked to enter your mood on a 1–10 scale (1=worst, 10=best). 

The study seeks to understand links between sensor data and well-being and to support caregivers during and after transplant. 

Each time step in this simulation represents one day.

Below is your background and history with the program. Based on this information, decide whether you will complete today’s mood survey.

- **Your Background:**
- You are in the 61+ age group.
- Your gender is Male.
- Your monthly income is $3,000 - $4,999.
- You are assigned to the Intervention arm of the study.
- Your role is Patients.
- Your transplant type is Allogeneic.
- You have stayed in the hospital for 20.
- You receive caregiving for Less than or equal to 40 hours per week.

- **Your caregiver partner’s background information**
- You are in the 61+ age group.
- Your gender is Female.
- Your monthly income is $5,000 - $6,999.
- You are assigned to the Intervention arm of the study.
- Your role is Caregivers.
- You are caring for a patient whose transplant type is Allogeneic.
- The patient you care for has stayed in the hospital for 20.
- You provide caregiving for Less than or equal to 40 hours per week.

- **Context: past behavior:**
Below is a record of your previous mood survey record (each representing one day):
{
  "Day 0": "Missing",
  "Day 1": "Missing",
  "Day 2": "Missing",
  "Day 3": "Missing",
  "Day 4": "Missing",
  "Day 5": 7.0,
  ...


```
  

### 3️⃣ Inference 
**Goal**
For each sampled participant and each target day, call the Azure OpenAI API with:
- The ```fixed system prompt``` describing the ROADMAP 2.0 study and the modeling goal
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
- The ```participant-specific user prompt``` from ```/prompts```

Then store the model’s ```.json``` response for analysis.

**Behavior**:
For a given ```(role, day)``` pair:

- Load ```prompts/{role}_day{day}.json```.

- Iterate over all participants in that file.

  - For each participant:
  
    - Send a chat completion request with:
  
      - ```Fixed system prompt```
  
      - ```participant-specific user prompt```

**Reply example**
Expect a JSON-formatted assistant reply with keys: ```id```, ```day```, ```reason_do```, ```reason_not```, ```decision```, ```confidence```:
bash```
"model_response_parsed": {
      "id": "P273",
      "day": 20,
      "reason_do": "I want to contribute to the study and check in now that my partner is home after a long 15-day hospital stay to help track how I'm coping.",
      "reason_not": "I haven't opened the app for any of the past 20 days and it's easy to forget or get busy with caregiving tasks and limited time/resources.",
      "decision": "No",
      "confidence": 0.88
    }
```







