#!/usr/bin/env python3
"""ESG Risk ML Pipeline.

Generates a synthetic ESG dataset, trains a Random Forest classifier,
and evaluates it with SHAP explainability, sector sub-models, and
hyperparameter tuning.

Usage:
    python main.py                  # train + evaluate
    python main.py --export         # also save dataset and predictions to CSV
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

SEED = 42
N_COMPANIES = 1_000
SECTORS = ["Manufacturing", "Technology", "Energy", "Finance", "Healthcare"]
FEATURES = [
    "carbon_intensity", "renewable_pct", "water_intensity",
    "waste_recycling",  "env_fines",
    "injury_rate",      "turnover_pct",  "gender_pay_gap",  "supply_chain_audit",
    "board_independence", "ceo_pay_ratio", "data_breaches",  "anti_corruption",
]
HIGHER_IS_BETTER = {
    "waste_recycling", "renewable_pct",
    "supply_chain_audit", "board_independence",
}


# ── Data generation ──────────────────────────────────────────────────────────

def generate_dataset(n: int = N_COMPANIES, seed: int = SEED) -> pd.DataFrame:
    """Create a synthetic ESG dataset with realistic sector-level variation."""
    rng = np.random.default_rng(seed)
    sectors = rng.choice(SECTORS, size=n)

    sector_carbon     = {"Manufacturing": 400, "Energy": 600, "Technology": 150,
                         "Finance": 80,  "Healthcare": 120}
    sector_renewables = {"Manufacturing": 0.20, "Energy": 0.15, "Technology": 0.45,
                         "Finance": 0.30, "Healthcare": 0.35}

    carbon         = np.array([rng.normal(sector_carbon[s], 80) for s in sectors]).clip(10, 900)
    renewable_pct  = np.array([rng.normal(sector_renewables[s], 0.10) for s in sectors]).clip(0, 1)

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
        "carbon_intensity":   carbon,
        "renewable_pct":      renewable_pct,
        "water_intensity":    water_intensity,
        "waste_recycling":    waste_recycling,
        "env_fines":          env_fines,
        "injury_rate":        injury_rate,
        "turnover_pct":       turnover_pct,
        "gender_pay_gap":     gender_pay_gap,
        "supply_chain_audit": supply_chain_audit,
        "board_independence": board_independence,
        "ceo_pay_ratio":      ceo_pay_ratio,
        "data_breaches":      data_breaches,
        "anti_corruption":    anti_corruption,
    })
    return _add_risk_labels(df)


def _add_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    e = (
        _norm(df["carbon_intensity"])    * 0.30 +
        _norm_inv(df["renewable_pct"])   * 0.25 +
        _norm(df["water_intensity"])     * 0.20 +
        _norm_inv(df["waste_recycling"]) * 0.15 +
        _norm(df["env_fines"])           * 0.10
    )
    s = (
        _norm(df["injury_rate"])            * 0.35 +
        _norm(df["turnover_pct"])           * 0.25 +
        _norm(df["gender_pay_gap"])         * 0.20 +
        _norm_inv(df["supply_chain_audit"]) * 0.20
    )
    g = (
        _norm_inv(df["board_independence"]) * 0.35 +
        _norm(df["ceo_pay_ratio"])          * 0.30 +
        _norm(df["data_breaches"])          * 0.25 +
        _norm_inv(df["anti_corruption"])    * 0.10
    )
    composite = 0.40 * e + 0.35 * s + 0.25 * g
    df["esg_score"]  = composite.round(4)
    df["risk_label"] = pd.qcut(composite, q=[0, 0.33, 0.66, 1.0],
                                labels=["Low", "Medium", "High"])
    return df


def _norm(s: pd.Series) -> pd.Series:
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

def _norm_inv(s: pd.Series) -> pd.Series:
    return 1 - _norm(s)


# ── Hyperparameter tuning ────────────────────────────────────────────────────

PARAM_GRID = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth":    [8, 12, None],
    "clf__min_samples_leaf": [1, 3],
}


def tune_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Run GridSearchCV and return the best-fitted pipeline."""
    print("\n[1/4] Hyperparameter tuning (GridSearchCV, 3-fold CV)...")
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(class_weight="balanced",
                                       random_state=SEED, n_jobs=-1)),
    ])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    search = GridSearchCV(base, PARAM_GRID, cv=cv, scoring="f1_macro",
                          n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    print(f"    Best params : {search.best_params_}")
    print(f"    Best CV F1  : {search.best_score_:.3f}")
    return search.best_estimator_


# ── Global model evaluation ──────────────────────────────────────────────────

def evaluate_global(model: Pipeline, X_test: pd.DataFrame,
                    y_test: pd.Series) -> None:
    print("\n[2/4] Global model evaluation...")
    y_pred = model.predict(X_test)
    acc    = (y_pred == y_test).mean()
    cm     = confusion_matrix(y_test, y_pred, labels=["Low", "Medium", "High"])

    print(f"\n    Test accuracy : {acc:.1%}  ({len(y_test)} samples)")
    print()
    print(classification_report(y_test, y_pred, labels=["Low", "Medium", "High"],
                                 target_names=["Low", "Medium", "High"]))

    print("    Confusion matrix (rows=actual, cols=predicted):")
    print("                Low   Med  High")
    for label, row in zip(["Low   ", "Medium", "High  "], cm):
        print(f"      {label}   {row[0]:4d}  {row[1]:4d}  {row[2]:4d}")

    clf = model.named_steps["clf"]
    imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\n    Feature importance (top 8):")
    for feat, val in imp.head(8).items():
        bar = "#" * int((val / imp.iloc[0]) * 28)
        print(f"      {feat:<25s}  {bar:<28s}  {val:.3f}")


# ── SHAP explainability ──────────────────────────────────────────────────────

def explain_with_shap(model: Pipeline, X_train: pd.DataFrame,
                      X_test: pd.DataFrame) -> None:
    print("\n[3/4] SHAP explainability (TreeExplainer)...")
    clf       = model.named_steps["clf"]
    scaler    = model.named_steps["scaler"]
    X_tr_sc   = pd.DataFrame(scaler.transform(X_train), columns=FEATURES)
    X_te_sc   = pd.DataFrame(scaler.transform(X_test),  columns=FEATURES)

    explainer  = shap.TreeExplainer(clf, X_tr_sc)
    shap_vals  = explainer(X_te_sc)          # shape: (n_samples, n_features, n_classes)

    classes = list(clf.classes_)
    print()
    for i, cls in enumerate(classes):
        mean_abs = np.abs(shap_vals.values[:, :, i]).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[::-1][:5]
        print(f"    Top drivers for '{cls}' risk:")
        for j in top_idx:
            bar = "#" * int((mean_abs[j] / mean_abs[top_idx[0]]) * 24)
            print(f"      {FEATURES[j]:<25s}  {bar:<24s}  {mean_abs[j]:.4f}")
        print()


# ── Sector sub-models ────────────────────────────────────────────────────────

def train_sector_models(df: pd.DataFrame, global_acc: float) -> None:
    print("[4/4] Sector sub-models...")
    print(f"\n    {'Sector':<15s}  {'N':>5}  {'Accuracy':>9}  {'vs Global':>10}")
    print("    " + "-" * 46)

    for sector in sorted(SECTORS):
        sub = df[df["sector"] == sector]
        if len(sub) < 40:
            continue
        X_s = sub[FEATURES]
        y_s = sub["risk_label"]

        # Need at least 2 classes to train
        if y_s.nunique() < 2:
            continue

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_s, y_s, test_size=0.20, random_state=SEED, stratify=y_s
        )
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=10,
                                           class_weight="balanced",
                                           random_state=SEED, n_jobs=-1)),
        ])
        pipe.fit(X_tr, y_tr)
        acc   = (pipe.predict(X_te) == y_te).mean()
        delta = acc - global_acc
        sign  = "+" if delta >= 0 else ""
        print(f"    {sector:<15s}  {len(sub):>5}  {acc:>8.1%}  {sign}{delta:>8.1%}")
    print()


# ── CSV export ───────────────────────────────────────────────────────────────

def export_csv(df: pd.DataFrame, model: Pipeline) -> None:
    out = Path("output")
    out.mkdir(exist_ok=True)

    # Full dataset
    df.to_csv(out / "esg_dataset.csv", index=False)

    # Predictions on the full dataset
    proba   = model.predict_proba(df[FEATURES])
    classes = model.classes_
    pred_df = pd.DataFrame({
        "sector":         df["sector"].values,
        "esg_score":      df["esg_score"].values,
        "actual_label":   df["risk_label"].values,
        "predicted_label": model.predict(df[FEATURES]),
        **{f"prob_{c}": proba[:, i] for i, c in enumerate(classes)},
    })
    pred_df.to_csv(out / "predictions.csv", index=False)

    print(f"    Exported to {out.resolve()}/")
    print(f"      esg_dataset.csv   ({len(df)} rows)")
    print(f"      predictions.csv   ({len(pred_df)} rows)")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ESG Risk ML Pipeline")
    parser.add_argument("--export", action="store_true",
                        help="Save dataset and predictions to output/ as CSV")
    args = parser.parse_args()

    print("=" * 60)
    print("  ESG Risk ML Pipeline")
    print("=" * 60)

    print("\nGenerating synthetic ESG dataset...")
    df   = generate_dataset()
    dist = df["risk_label"].value_counts().to_dict()
    print(f"  {len(df)} companies  |  "
          f"Low: {dist.get('Low',0)}  "
          f"Medium: {dist.get('Medium',0)}  "
          f"High: {dist.get('High',0)}")

    X = df[FEATURES]
    y = df["risk_label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    # 1. Hyperparameter tuning
    model = tune_model(X_train, y_train)

    # 2. Global evaluation
    evaluate_global(model, X_test, y_test)
    global_acc = (model.predict(X_test) == y_test).mean()

    # 3. SHAP explainability
    explain_with_shap(model, X_train, X_test)

    # 4. Sector sub-models
    train_sector_models(df, global_acc)

    # Optional CSV export
    if args.export:
        print("Exporting CSVs...")
        export_csv(df, model)

    print("=" * 60)


if __name__ == "__main__":
    main()
