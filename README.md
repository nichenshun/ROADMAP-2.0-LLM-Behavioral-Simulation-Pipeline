# ROADMAP-2.0-LLM-Behavioral-Simulation-Pipeline

**Simulation and prediction pipeline for participant behavior in the ROADMAP 2.0 mobile-health study using large language models (LLMs).**

This repository implements an end-to-end experimental framework for simulating and predicting participant mood-survey completion behavior in the *ROADMAP 2.0* clinical study.  
The project integrates data preprocessing, prompt generation, Azure OpenAI inference, and evaluation modules—enabling reproducible behavioral modeling and analysis of both caregivers and patients.

---

## 📂 Project Overview

### 🔹 Modules
| Script | Description |
|--------|--------------|
| `data_process.py` | Merges all study CSVs under `/data/` into a unified `participants_demographics.json`, formatting behavioral and demographic records. |
| `prompts_gen.py` | Randomly samples **100 patients** and **100 caregivers**, then generates role-specific prompts for prediction days (0, 20, 40, 60, 80, 100, 120). |
| `inference.py` | Calls **Azure OpenAI API** to simulate participant decisions (Yes/No) for mood-survey completion, saving predictions incrementally to JSON. |
| `eval.ipynb` | Evaluates model predictions vs. ground truth (from `participants_demographics.json`) and visualizes metrics such as TPR, FPR, accuracy, and cross-entropy loss. |

---

## 🧩 Pipeline Steps

### 1️⃣ Data Processing
Run the preprocessing script to integrate demographic, mood, and behavioral data:
```bash
python data_process.py
```

This produces a comprehensive file: 
``` bash
participants_demographics.json
```
where each participant record includes:

Demographics

Daily steps, sedentary minutes, active minutes

Sleep duration 

Mood-survey history (Days 0–120)

### 2️⃣ Prompt Generation
Generate LLM simulation prompts for both participant groups:
```bash
python prompts_gen.py
```

This will create prompt files under ```/prompts/```, e.g.:
```bash
prompts/
├── patients_day0.json
├── patients_day20.json
...
├── caregivers_day120.json
```

Each file contains 100 randomy sampled participants per caregivers and patients.

### 3️⃣ Inference via Azure OpenAI
Run behavioral simulations using Azure OpenAI API:
```bash
python inference.py
```

The script iterates through all participant prompts for the specified day and role,
calls the model (e.g., gpt-5-mini), and saves each result immediately to /predictions/.




