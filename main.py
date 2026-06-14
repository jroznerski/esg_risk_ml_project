#!/usr/bin/env python3
"""ESG Risk ML Pipeline — no API required.

Generates a synthetic ESG dataset, trains a Random Forest classifier,
and evaluates it with feature importance analysis.

Usage:
    python main.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

SEED = 42
N_COMPANIES = 1_000

SECTORS = ["Manufacturing", "Technology", "Energy", "Finance", "Healthcare"]

# Risk label thresholds (applied to weighted ESG score)
RISK_LABELS = {
    (0.00, 0.33): "Low",
    (0.33, 0.66): "Medium",
    (0.66, 1.00): "High",
}


# ── Data generation ──────────────────────────────────────────────────────────

def generate_dataset(n: int = N_COMPANIES, seed: int = SEED) -> pd.DataFrame:
    """Create a synthetic ESG dataset with realistic sector-level variation."""
    rng = np.random.default_rng(seed)

    sectors = rng.choice(SECTORS, size=n)

    # Sector baselines: higher value = worse metric (except where noted)
    sector_carbon = {"Manufacturing": 400, "Energy": 600, "Technology": 150,
                     "Finance": 80,  "Healthcare": 120}
    sector_renewables = {"Manufacturing": 0.20, "Energy": 0.15, "Technology": 0.45,
                         "Finance": 0.30, "Healthcare": 0.35}

    carbon = np.array([
        rng.normal(sector_carbon[s], 80) for s in sectors
    ]).clip(10, 900)

    renewable_pct = np.array([
        rng.normal(sector_renewables[s], 0.10) for s in sectors
    ]).clip(0, 1)

    water_intensity    = rng.normal(250, 70, n).clip(20, 700)
    waste_recycling    = rng.uniform(0.10, 0.90, n)
    env_fines          = rng.poisson(1.2, n).clip(0, 10)

    injury_rate        = rng.exponential(2.5, n).clip(0, 15)
    turnover_pct       = rng.normal(0.18, 0.07, n).clip(0.02, 0.60)
    gender_pay_gap     = rng.normal(0.12, 0.06, n).clip(0, 0.40)
    supply_chain_audit = rng.uniform(0.20, 1.00, n)

    board_independence = rng.normal(0.65, 0.12, n).clip(0.20, 1.00)
    ceo_pay_ratio      = rng.lognormal(5.2, 0.5, n).clip(20, 800)
    data_breaches      = rng.poisson(0.5, n).clip(0, 5)
    anti_corruption    = rng.choice([0, 1], size=n, p=[0.15, 0.85])

    df = pd.DataFrame({
        "sector":             sectors,
        # Environmental
        "carbon_intensity":   carbon,
        "renewable_pct":      renewable_pct,
        "water_intensity":    water_intensity,
        "waste_recycling":    waste_recycling,
        "env_fines":          env_fines,
        # Social
        "injury_rate":        injury_rate,
        "turnover_pct":       turnover_pct,
        "gender_pay_gap":     gender_pay_gap,
        "supply_chain_audit": supply_chain_audit,
        # Governance
        "board_independence": board_independence,
        "ceo_pay_ratio":      ceo_pay_ratio,
        "data_breaches":      data_breaches,
        "anti_corruption":    anti_corruption,
    })

    df = _add_risk_labels(df)
    return df


def _add_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a weighted ESG risk score and assign Low / Medium / High labels."""
    # Normalise each metric to [0, 1] where 1 = most risky
    e = (
        _norm(df["carbon_intensity"])   * 0.30 +
        _norm_inv(df["renewable_pct"])  * 0.25 +
        _norm(df["water_intensity"])    * 0.20 +
        _norm_inv(df["waste_recycling"])* 0.15 +
        _norm(df["env_fines"])          * 0.10
    )
    s = (
        _norm(df["injury_rate"])           * 0.35 +
        _norm(df["turnover_pct"])          * 0.25 +
        _norm(df["gender_pay_gap"])        * 0.20 +
        _norm_inv(df["supply_chain_audit"])* 0.20
    )
    g = (
        _norm_inv(df["board_independence"]) * 0.35 +
        _norm(df["ceo_pay_ratio"])          * 0.30 +
        _norm(df["data_breaches"])          * 0.25 +
        _norm_inv(df["anti_corruption"])    * 0.10
    )

    # Weighted composite: E 40%, S 35%, G 25%
    composite = 0.40 * e + 0.35 * s + 0.25 * g

    df["esg_score"] = composite.round(4)
    # Use quantile-based bins so all three classes are always populated
    df["risk_label"] = pd.qcut(
        composite,
        q=[0, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"],
    )
    return df


def _norm(s: pd.Series) -> pd.Series:
    """Min-max normalise (higher raw → higher risk)."""
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def _norm_inv(s: pd.Series) -> pd.Series:
    """Min-max normalise inverted (lower raw → higher risk)."""
    return 1 - _norm(s)


# ── ML pipeline ──────────────────────────────────────────────────────────────

FEATURES = [
    "carbon_intensity", "renewable_pct", "water_intensity",
    "waste_recycling",  "env_fines",
    "injury_rate",      "turnover_pct",  "gender_pay_gap",  "supply_chain_audit",
    "board_independence", "ceo_pay_ratio", "data_breaches",  "anti_corruption",
]


def build_and_evaluate(df: pd.DataFrame) -> None:
    X = df[FEATURES]
    y = df["risk_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    _print_results(model, X_test, y_test, y_pred)
    _print_feature_importance(model)
    _print_sample_predictions(model, df)


def _print_results(model, X_test, y_test, y_pred) -> None:
    acc = (y_pred == y_test).mean()
    cm  = confusion_matrix(y_test, y_pred, labels=["Low", "Medium", "High"])

    print("\n" + "=" * 60)
    print("  ESG RISK CLASSIFICATION - RESULTS")
    print("=" * 60)
    print(f"\n  Test accuracy : {acc:.1%}")
    print(f"  Test samples  : {len(y_test)}")
    print()
    print(classification_report(y_test, y_pred, labels=["Low", "Medium", "High"]))

    print("  Confusion matrix (rows=actual, cols=predicted):")
    print(f"               Low   Med   High")
    labels = ["Low   ", "Medium", "High  "]
    for label, row in zip(labels, cm):
        print(f"    {label}    {row[0]:4d}  {row[1]:4d}  {row[2]:4d}")


def _print_feature_importance(model) -> None:
    clf = model.named_steps["clf"]
    importances = pd.Series(clf.feature_importances_, index=FEATURES)
    top = importances.sort_values(ascending=False).head(8)

    print("\n  Top 8 features by importance:")
    bar_max = top.max()
    for feat, imp in top.items():
        bar_len = int((imp / bar_max) * 30)
        print(f"    {feat:<25s}  {'#' * bar_len}  {imp:.3f}")


def _print_sample_predictions(model, df: pd.DataFrame) -> None:
    sample = df[FEATURES].sample(5, random_state=SEED)
    preds  = model.predict(sample)
    proba  = model.predict_proba(sample)
    actual = df.loc[sample.index, "risk_label"].values

    print("\n  Sample predictions (5 random companies):")
    print(f"    {'Actual':<10}  {'Predicted':<10}  {'Low%':>6}  {'Med%':>6}  {'High%':>6}")
    print("    " + "-" * 50)
    classes = model.classes_
    low_i  = list(classes).index("Low")
    med_i  = list(classes).index("Medium")
    high_i = list(classes).index("High")
    for act, pred, p in zip(actual, preds, proba):
        match = "OK" if act == pred else "!!"
        print(
            f"    {str(act):<10}  {str(pred):<10}  "
            f"{p[low_i]:>5.0%}  {p[med_i]:>5.0%}  {p[high_i]:>5.0%}  {match}"
        )
    print("\n" + "=" * 60 + "\n")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating synthetic ESG dataset...")
    df = generate_dataset()

    dist = df["risk_label"].value_counts().to_dict()
    print(f"Dataset: {len(df)} companies  |  Low: {dist.get('Low',0)}  "
          f"Medium: {dist.get('Medium',0)}  High: {dist.get('High',0)}")

    print("Training Random Forest classifier...")
    build_and_evaluate(df)
