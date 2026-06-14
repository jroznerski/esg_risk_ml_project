# ESG Risk ML Project

A machine learning pipeline that classifies companies into **Low**, **Medium**, and **High** ESG risk categories using a Random Forest classifier.

## What it does

1. **Generates** a synthetic dataset of 1,000 companies across 5 sectors (Manufacturing, Technology, Energy, Finance, Healthcare) with realistic ESG metrics
2. **Scores** each company using a weighted composite of Environmental (40%), Social (35%), and Governance (25%) pillars
3. **Trains** a Random Forest classifier on 13 ESG features
4. **Evaluates** the model with accuracy, classification report, confusion matrix, feature importance, and sample predictions

## Results

- **73.5% test accuracy** on 200 held-out companies
- Low and High risk classes achieve F1 ~0.80; Medium is harder to separate (F1 0.59)
- Top predictive features: carbon intensity, renewable energy %, supply chain audit coverage, injury rate

## Features used

| Pillar | Features |
|---|---|
| Environmental | Carbon intensity, renewable energy %, water intensity, waste recycling rate, environmental fines |
| Social | Injury rate, employee turnover, gender pay gap, supply chain audit coverage |
| Governance | Board independence, CEO pay ratio, data breaches, anti-corruption policy |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install scikit-learn pandas numpy
```

## Run

```bash
python main.py
```

## Extending it

- Replace the synthetic data generator with real ESG data (MSCI, Sustainalytics, Bloomberg)
- Add SHAP values for per-company explainability
- Train sector-specific sub-models
- Export predictions to CSV with `--export` flag
