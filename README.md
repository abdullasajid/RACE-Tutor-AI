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

You can also load and flatten the HuggingFace `ehovy/race` splits directly while training:

```bash
python -m backend.train_model_a --source huggingface --save-flattened
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Included Artifacts

This repository includes the files needed for a working demo:

```text
models/model_a/logreg_candidate_rich.pkl   # best trained Model A
data/raw/test.csv                          # demo RACE split for random samples
report/main.pdf                            # final report
presentation/slides.pdf                    # final presentation deck
```

The full training split is not required to run the app. Add `data/raw/train.csv`
and `data/raw/val.csv` only if you want to retrain Model A.

## Train Model A

```bash
python -m backend.train_model_a --train data/raw/train.csv --val data/raw/val.csv
```

The strongest current Model A trains one candidate example per answer option and
uses binary bag-of-words, bigrams, TF-IDF similarity, overlap, negation, and
per-question relative option features:

```bash
python -m backend.train_model_a \
  --train data/raw/train.csv \
  --val data/raw/val.csv \
  --test data/raw/test.csv \
  --model logreg_candidate \
  --class-weight none \
  --ngram-max 2 \
  --c 0.3 \
  --output-name logreg_candidate_rich
```

## Run App

```bash
streamlit run frontend/app.py
```

The app uses:

- `models/model_a/logreg_candidate_rich.pkl` for answer verification.
- `backend/model_b.py` for generated distractors and graduated hints.

## Tests and Smoke Checks

```bash
python -m pytest -q
python -m compileall backend frontend/app.py tests
```

If `pytest` is missing, install dependencies again with `pip install -r requirements.txt`.

## Reports and Slides

Build the final report:

```bash
pdflatex -interaction=nonstopmode -halt-on-error -output-directory report report/main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory report report/main.tex
```

Build the presentation:

```bash
pdflatex -interaction=nonstopmode -halt-on-error -output-directory presentation presentation/slides.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory presentation presentation/slides.tex
```

## Model B

Model B is a deterministic classical component. It extracts passage phrases for
distractors, filters out the correct answer, ranks candidates by passage
frequency and question relevance, and generates three hints from broad guidance
to direct supporting evidence.

## Main Components

- Model A: answer verification using traditional ML.
- Model B: distractor and hint generation using extractive classical methods.
- UI: article input, quiz view, hint panel, and analytics dashboard.
