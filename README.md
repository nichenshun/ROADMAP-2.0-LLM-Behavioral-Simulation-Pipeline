# ROADMAP-2.0-LLM-Behavioral-Simulation-Pipeline

**Simulation and prediction pipeline for participant behavior in the ROADMAP 2.0 mobile-health study using large language models (LLMs).**

This research explores the application of large language models (LLMs) to simulate long-term human behavioral trajectories in clinical settings. 

Our work investigates whether LLMs can serve as realistic behavioral simulators by modeling the day-to-day decision-making processes of participants in the ROADMAP 2.0 mobile-health study. Specifically, we focus on predicting daily mood survey completion behavior over a 120-day post-transplant period, examining how LLMs incorporate demographic characteristics, historical completion patterns, activity data, and situational factors to generate plausible behavioral sequences. 

The primary objectives are to:  

(1) assess the accuracy of LLM-simulated behavioral trajectories against ground truth data,  

(2) analyze performance variations across different demographic subgroups,  

(3) establish baseline comparisons with traditional majority-vote prediction methods. 

---

## 📂 Project Overview

### 🔹 Modules
| Script | Description |
|--------|--------------|
| ```data_process.py``` | Merges some of the study ```.csv```'s under ```/data/``` into a unified ```participants_data.json```, formatting behavioral and demographic records. |
| ```/prompt``` | generates role-specific (patient & caregiver) prompts for prediction. |
| ```/inference``` | Calls **Azure OpenAI API** to simulate participant decisions (Yes/No) for mood-survey completion, saving predictions incrementally to ```.json```. |
| ```/eval``` | Evaluates model predictions vs. ground truth (from ```participants_data.json```) and visualizes metrics such as TPR, FPR, overall accuracy, and cross-entropy loss. |

---
## 🧩 Pipeline Steps
### 1️⃣ Data Processing 
**Goal**

Build a unified `.json` dataset `participants_data.json` from multiple raw `.csv` files under `/data`.
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

- User prompt for patients (e.g., ```"P117"```, ```target=Day 40```):
```bash
You are a patient participant in a dyad in the ROADMAP 2.0 mobile-health study. 

You are a patient recovering from a hematopoietic cell transplantation (HCT) to treat a blood-related disease such as leukemia or lymphoma. During recovery, you may experience fatigue, nausea, pain, sleep disturbances, and emotional stress, while relying on your care partner and medical team to help you regain strength and prevent complications like infections or graft-versus-host disease. 

Each day, you are asked to enter your mood on a 1-10 scale (1=worst, 10=best). 

The study seeks to understand links between sensor data and well-being and to support caregivers during and after transplant. 

Each time step in this simulation represents one day.

Below is your background and history with the program. Based on this information, decide whether you will complete today's mood survey.

- **Your Background:**
- You are in the 61+ age group.
- Your gender is female.
- Your monthly income is $3,000 - $4,999.
- You are assigned to the Intervention arm of the study.
- Your transplant type is Autologous.
- You stayed in the hospital for 14 days.
- You receive caregiving from your caregiver for Less than or equal to 40 hours per week.

- **Your caregiver partner's background information**
- Your caregiver is in the 61+ age group.
- Your caregiver's gender is male.

- **Context: past behavior:**
Below is a record of your previous mood survey record (each representing one day):
{
  "Day 0": "missing",
  "Day 1": 8.0,
  "Day 2": "missing",
  "Day 3": "missing",
  "Day 4": "missing",
  "Day 5": 1.0,
  "Day 6": 8.0,
  "Day 7": 9.0,
  "Day 8": 9.0,
  "Day 9": 9.0,
  "Day 10": "missing",
  "Day 11": 9.0,
  "Day 12": "missing",
  "Day 13": "missing",
  "Day 14": 10.0,
  "Day 15": "missing",
  "Day 16": "missing",
  "Day 17": "missing",
  "Day 18": "missing",
  "Day 19": "missing",
  "Day 20": "missing",
  "Day 21": "missing",
  "Day 22": "missing",
  "Day 23": "missing",
  "Day 24": "missing",
  "Day 25": "missing",
  "Day 26": "missing",
  "Day 27": "missing",
  "Day 28": "missing",
  "Day 29": "missing",
  "Day 30": "missing",
  "Day 31": "missing",
  "Day 32": "missing",
  "Day 33": "missing",
  "Day 34": "missing",
  "Day 35": "missing",
  "Day 36": "missing",
  "Day 37": "missing",
  "Day 38": "missing",
  "Day 39": "missing"
}

- **Question:** Today is Day 40. Will you complete today's mood survey?
- **Explain:** Give one reason to complete the survey and one reason not to; then state your final decision for today (Yes / No).
- **Estimate:** Report the probability that your decision is correct as a single number between 0 and 1 (e.g., 0.73).

At the end, output a single JSON object with exactly these keys (no extra text):
{"id": "P117", "day": 40, "reason_do": "<string>", "reason_not": "<string>", "decision": "<Yes|No>", "confidence": <float between 0 and 1>}

```

User prompt for caregivers (e.g., ```"P210"```, ```target=Day 60```):
```bash
You are a caregiver participant in a dyad in the ROADMAP 2.0 mobile-health study. 

You are a caregiver for a loved one recovering from a hematopoietic cell transplantation (HCT) for a serious blood or immune disorder. You provide daily physical and emotional support, monitor for complications, manage medications and appointments, and balance these duties with your own daily work routine. 

Each day, you are asked to enter your mood on a 1-10 scale (1=worst, 10=best). 

The study seeks to understand links between sensor data and well-being and to support caregivers during and after transplant. 

Each time step in this simulation represents one day.

Below is your background and history with the program. Based on this information, decide whether you will complete today's mood survey.

- **Your Background:**
- You are in the 61+ age group.
- Your gender is female.
- Your monthly income is Less than $1,000.
- You provide caregiving to your patient for Less than or equal to 40 hours per week.

- **Your patient partner's background information**
- Your patient is in the 61+ age group.
- Your patient's gender is male.
- Your patient is assigned to the Intervention arm of the study.
- Your patient's transplant type is Allogeneic.
- Your patient is an OHSU patient.

- **Context: past behavior:**
Below is a record of your previous mood survey record (each representing one day):
{
  "Day 0": "missing",
  "Day 1": 9.0,
  "Day 2": 9.0,
  "Day 3": 9.0,
  "Day 4": 9.0,
  "Day 5": 9.0,
  "Day 6": 9.0,
  "Day 7": 8.0,
  "Day 8": 9.0,
  "Day 9": 9.0,
  "Day 10": 8.0,
  "Day 11": 9.0,
  "Day 12": 9.0,
  "Day 13": 9.0,
  "Day 14": 9.0,
  "Day 15": 9.0,
  "Day 16": 10.0,
  "Day 17": 9.0,
  "Day 18": 9.0,
  "Day 19": 9.0,
  "Day 20": 9.0,
  "Day 21": 9.0,
  "Day 22": 9.0,
  "Day 23": 9.0,
  "Day 24": 9.0,
  "Day 25": "missing",
  "Day 26": 9.0,
  "Day 27": "missing",
  "Day 28": 9.0,
  "Day 29": 8.0,
  "Day 30": 9.0,
  "Day 31": 9.0,
  "Day 32": 6.0,
  "Day 33": 9.0,
  "Day 34": 9.0,
  "Day 35": 9.0,
  "Day 36": "missing",
  "Day 37": 8.0,
  "Day 38": 9.0,
  "Day 39": 9.0,
  "Day 40": 9.0,
  "Day 41": "missing",
  "Day 42": 9.0,
  "Day 43": 9.0,
  "Day 44": 9.0,
  "Day 45": 9.0,
  "Day 46": 9.0,
  "Day 47": 9.0,
  "Day 48": 9.0,
  "Day 49": 7.0,
  "Day 50": 7.0,
  "Day 51": 8.0,
  "Day 52": 8.0,
  "Day 53": 9.0,
  "Day 54": 9.0,
  "Day 55": 9.0,
  "Day 56": "missing",
  "Day 57": 9.0,
  "Day 58": 9.0,
  "Day 59": 9.0
}

- **Question:** Today is Day 60. Will you complete today's mood survey?
- **Explain:** Give one reason to complete the survey and one reason not to; then state your final decision for today (Yes / No).
- **Estimate:** Report the probability that your decision is correct as a single number between 0 and 1 (e.g., 0.73).

At the end, output a single JSON object with exactly these keys (no extra text):
{"id": "P210", "day": 60, "reason_do": "<string>", "reason_not": "<string>", "decision": "<Yes|No>", "confidence": <float between 0 and 1>}
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

**Behavior**

For a given ```(role, day)``` pair:

- Load ```prompts/{role}_day{day}.json```.

- Iterate over all participants in that file.

  - For each participant:
  
    - Send a chat completion request with:
  
      - ```Fixed system prompt```
  
      - ```participant-specific user prompt```
     
**Run**
- Patient, Day 20:
```bash
python inference.py patients 20 \
  --api-key "YOUR_AZURE_OPENAI_API_KEY" \
  --endpoint "https://YOUR-RESOURCE.openai.azure.com/" \
  --deployment "YOUR_DEPLOYMENT_NAME"
```
- Caregiver, Day 60:
```bash
python inference.py caregivers 60 \
  --api-key "YOUR_AZURE_OPENAI_API_KEY" \
  --endpoint "https://YOUR-RESOURCE.openai.azure.com/" \
  --deployment "YOUR_DEPLOYMENT_NAME"
```

**Reply example**

Expect a JSON-formatted assistant reply with keys: ```id```, ```day```, ```reason_do```, ```reason_not```, ```decision```, ```confidence```:
```bash
"model_response_parsed": {
      "id": "P273",
      "day": 20,
      "reason_do": "I want to contribute to the study and check in now that my partner is home after a long 15-day hospital stay to help track how I'm coping.",
      "reason_not": "I haven't opened the app for any of the past 20 days and it's easy to forget or get busy with caregiving tasks and limited time/resources.",
      "decision": "No",
      "confidence": 0.88
    }
```







