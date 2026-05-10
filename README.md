# RACE Tutor AI

RACE Tutor AI is an AI-powered reading comprehension and quiz generation system built on the RACE dataset. It uses classical machine learning to verify answers, generate plausible distractors, provide graduated hints, and present the full quiz experience through a Streamlit interface.

## Project Structure

```text
RACE-Tutor-AI/
├── frontend/              # Streamlit user interface
├── backend/               # ML, preprocessing, inference, evaluation
├── data/
│   ├── raw/               # train.csv, val.csv, test.csv
│   └── processed/         # generated processed files
├── models/
│   ├── model_a/           # answer verifier models
│   └── model_b/           # distractor/hint models or artifacts
├── notebooks/             # EDA and experiments
├── report/                # final report
├── presentation/          # slides or demo notes
└── tests/                 # unit tests
```

## Dataset

Place the RACE CSV files here:

```text
data/raw/train.csv
data/raw/val.csv
data/raw/test.csv
```

Expected columns:

```text
id, article, question, A, B, C, D, answer
```

If you have one combined CSV, create the required 80/10/10 split with:

```bash
python -m backend.split_dataset --input data/raw/race.csv
```

This writes `data/raw/train.csv`, `data/raw/val.csv`, and `data/raw/test.csv`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train Model A

```bash
python -m backend.train_model_a --train data/raw/train.csv --val data/raw/val.csv
```

## Run App

```bash
streamlit run frontend/app.py
```

## Main Components

- Model A: answer verification using traditional ML.
- Model B: distractor and hint generation using extractive classical methods.
- UI: article input, quiz view, hint panel, and analytics dashboard.
