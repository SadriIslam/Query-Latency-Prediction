"""
ml_model.py
-----------
Project: Predicting Database Query Latency for Medical Analytics Applications
Course:  CS 8260 - Advanced Database Systems, KSU Spring 2026
Team:    Sadri Islam, Kazi Shaharair Sharif, Tahsin Tasnia Khan

This script:
  1. Loads the 60-query benchmark dataset
  2. Trains 3 ML models: Random Forest, Gradient Boosting, SVR
  3. Compares them against PostgreSQL planner as baseline
  4. Prints results table + hypothesis test
  5. Saves a 4-panel results chart as results_chart.png

Usage:
  python ml_model.py

Requirements:
  pip install pandas numpy scikit-learn matplotlib
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')


# ── Configuration ──────────────────────────────────────────────────────
DATA_PATH    = os.path.join(os.path.dirname(__file__), '..', 'data', 'benchmark_dataset_60.csv')
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'results')
CHART_PATH   = os.path.join(RESULTS_DIR, 'results_chart.png')
RANDOM_SEED  = 42
TEST_SIZE    = 0.20

FEATURE_COLS = [
    "est_cost",       # PostgreSQL estimated cost (dimensionless)
    "est_rows",       # Estimated rows to process
    "num_joins",      # Number of table joins
    "join_type",      # Hash=1, Nested Loop=2, Merge=3
    "scan_type",      # Sequential=1, Index=2, Bitmap=3
    "planning_time",  # Time PostgreSQL spent planning (ms)
]
TARGET_COL = "execution_time"  # Actual execution time in ms — what we predict


# ── Load Data ──────────────────────────────────────────────────────────
def load_data():
    print("=" * 60)
    print("  LOADING DATASET")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"\n  Queries loaded    : {len(df)}")
    print(f"  Latency range     : {df[TARGET_COL].min():.3f} ms  to  {df[TARGET_COL].max():.3f} ms")
    print(f"  Mean latency      : {df[TARGET_COL].mean():.2f} ms")

    fast   = len(df[df[TARGET_COL] < 5])
    medium = len(df[(df[TARGET_COL] >= 5) & (df[TARGET_COL] < 100)])
    slow   = len(df[(df[TARGET_COL] >= 100) & (df[TARGET_COL] < 500)])
    vslow  = len(df[df[TARGET_COL] >= 500])

    print(f"\n  Fast    (< 5ms)   : {fast} queries")
    print(f"  Medium  (5-100ms) : {medium} queries")
    print(f"  Slow  (100-500ms) : {slow} queries")
    print(f"  V.Slow  (500ms+)  : {vslow} queries")

    return df


# ── Split Data ─────────────────────────────────────────────────────────
def split_data(df):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    # Standardize features for SVR
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"\n  Training queries  : {len(X_train)}")
    print(f"  Test queries      : {len(X_test)}")

    return X, y, X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, scaler


# ── Train Models ───────────────────────────────────────────────────────
def train_models(X, y, X_train, X_test, y_train, y_test, X_train_sc, X_test_sc):
    print("\n" + "=" * 60)
    print("  TRAINING MODELS")
    print("=" * 60)

    # ── Random Forest ─────────────────────────────────────────────────
    # Builds 200 independent decision trees on random samples.
    # Averages all predictions to reduce variance.
    print("\n  [1/3] Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators   = 200,
        max_depth      = 8,
        min_samples_split = 3,
        random_state   = RANDOM_SEED
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_mae   = mean_absolute_error(y_test, rf_preds)
    rf_rmse  = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_cv    = -cross_val_score(rf, X, y, cv=5, scoring='neg_mean_absolute_error').mean()
    print(f"       MAE: {rf_mae:.3f} ms  |  RMSE: {rf_rmse:.3f} ms  |  CV-MAE: {rf_cv:.3f} ms")

    # ── Gradient Boosting ─────────────────────────────────────────────
    # Builds 300 trees sequentially — each corrects errors of the previous.
    # Learning rate of 0.05 keeps it from overfitting.
    print("\n  [2/3] Training Gradient Boosting...")
    gb = GradientBoostingRegressor(
        n_estimators   = 300,
        learning_rate  = 0.05,
        max_depth      = 4,
        subsample      = 0.8,
        min_samples_split = 3,
        random_state   = RANDOM_SEED
    )
    gb.fit(X_train, y_train)
    gb_preds = gb.predict(X_test)
    gb_mae   = mean_absolute_error(y_test, gb_preds)
    gb_rmse  = np.sqrt(mean_squared_error(y_test, gb_preds))
    gb_cv    = -cross_val_score(gb, X, y, cv=5, scoring='neg_mean_absolute_error').mean()
    print(f"       MAE: {gb_mae:.3f} ms  |  RMSE: {gb_rmse:.3f} ms  |  CV-MAE: {gb_cv:.3f} ms")

    # ── SVR ───────────────────────────────────────────────────────────
    # Finds a regression hyperplane with an RBF kernel.
    # Requires standardized features (done above with StandardScaler).
    print("\n  [3/3] Training SVR (RBF Kernel)...")
    svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=5)
    svr.fit(X_train_sc, y_train)
    svr_preds = svr.predict(X_test)
    svr_mae   = mean_absolute_error(y_test, svr_preds)
    svr_rmse  = np.sqrt(mean_squared_error(y_test, svr_preds))
    svr_cv    = -cross_val_score(
        SVR(kernel='rbf', C=100, gamma=0.1, epsilon=5),
        X_train_sc, y_train, cv=5,
        scoring='neg_mean_absolute_error'
    ).mean()
    print(f"       MAE: {svr_mae:.3f} ms  |  RMSE: {svr_rmse:.3f} ms  |  CV-MAE: {svr_cv:.3f} ms")

    # ── PostgreSQL Baseline ────────────────────────────────────────────
    # Scale estimated cost to milliseconds using a simple ratio.
    scale    = y.mean() / X["est_cost"].replace(0, 0.001).mean()
    pg_preds = X_test["est_cost"] * scale
    pg_mae   = mean_absolute_error(y_test, pg_preds)
    pg_rmse  = np.sqrt(mean_squared_error(y_test, pg_preds))

    models = {
        "PostgreSQL Planner": {"mae": pg_mae, "rmse": pg_rmse, "cv": None,   "preds": pg_preds},
        "Random Forest":      {"mae": rf_mae, "rmse": rf_rmse, "cv": rf_cv,  "preds": rf_preds},
        "Gradient Boosting":  {"mae": gb_mae, "rmse": gb_rmse, "cv": gb_cv,  "preds": gb_preds},
        "SVR (RBF Kernel)":   {"mae": svr_mae,"rmse": svr_rmse,"cv": svr_cv, "preds": svr_preds},
    }

    return models, gb


# ── Print Results ──────────────────────────────────────────────────────
def print_results(models, y_test, X, y, gb):
    print("\n" + "=" * 65)
    print("  FINAL RESULTS")
    print("=" * 65)
    print(f"\n  {'Model':<25} {'MAE (ms)':>10} {'RMSE (ms)':>10} {'CV-MAE':>10}")
    print("  " + "-" * 57)

    for name, m in models.items():
        cv_str = f"{m['cv']:.3f}" if m['cv'] else "N/A"
        print(f"  {name:<25} {m['mae']:>10.3f} {m['rmse']:>10.3f} {cv_str:>10}")

    # Hypothesis test
    pg_mae  = models["PostgreSQL Planner"]["mae"]
    ml_maes = {k: v["mae"] for k, v in models.items() if k != "PostgreSQL Planner"}
    best    = min(ml_maes, key=ml_maes.get)
    best_v  = ml_maes[best]

    print("\n" + "=" * 65)
    print("  HYPOTHESIS TEST")
    print("=" * 65)
    print(f"\n  PostgreSQL MAE : {pg_mae:.3f} ms")
    print(f"  Best ML MAE   : {best_v:.3f} ms  ({best})")

    if best_v < pg_mae:
        pct = (pg_mae - best_v) / pg_mae * 100
        print(f"\n  H1 SUPPORTED ✓")
        print(f"  EML ({best_v:.3f}) < EPG ({pg_mae:.3f})")
        print(f"  {best} is {pct:.1f}% more accurate than PostgreSQL!")
    else:
        print(f"\n  H0 holds — PostgreSQL planner is equal or better")

    # Feature importance
    print("\n" + "=" * 65)
    print("  FEATURE IMPORTANCE  (Gradient Boosting)")
    print("=" * 65)
    for feat, imp in sorted(zip(FEATURE_COLS, gb.feature_importances_),
                            key=lambda x: x[1], reverse=True):
        bar = "█" * int(imp * 50)
        print(f"  {feat:<15}  {imp:.4f}  {bar}")

    # Prediction details for best model
    print("\n" + "=" * 65)
    print(f"  {best.upper()} — PREDICTION vs ACTUAL")
    print("=" * 65)
    print(f"\n  {'Query':<6} {'Actual':>12} {'Predicted':>12} {'Error':>10}")
    print("  " + "-" * 44)
    best_preds = models[best]["preds"]
    for i, (actual, pred) in enumerate(zip(y_test.values, best_preds)):
        err = abs(actual - pred)
        print(f"  Q{i+1:<5} {actual:>12.3f} {pred:>12.3f} {err:>10.3f}")


# ── Save Chart ─────────────────────────────────────────────────────────
def save_chart(models, y_test, gb):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    gb_preds  = models["Gradient Boosting"]["preds"]
    rf_preds  = models["Random Forest"]["preds"]
    pg_preds  = models["PostgreSQL Planner"]["preds"]

    model_names  = ["PostgreSQL\nPlanner", "Random\nForest", "Gradient\nBoosting", "SVR\n(RBF)"]
    mae_vals     = [models[k]["mae"] for k in ["PostgreSQL Planner","Random Forest","Gradient Boosting","SVR (RBF Kernel)"]]
    rmse_vals    = [models[k]["rmse"] for k in ["PostgreSQL Planner","Random Forest","Gradient Boosting","SVR (RBF Kernel)"]]
    colors       = ["#666666","#0F4A3A","#1E3A5F","#3D1A6E"]
    edge_colors  = ["#444444","#0A3328","#152B47","#2E1252"]

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('white')
    fig.suptitle(
        "Project Update 3 — Advanced ML Query Latency Prediction\n60 Queries  |  3 Models  +  PostgreSQL Baseline",
        fontsize=14, fontweight="bold", y=0.98
    )

    bar_w = 0.55

    # MAE
    ax1 = fig.add_subplot(2, 2, 1)
    bars = ax1.bar(model_names, mae_vals, color=colors, edgecolor=edge_colors, width=bar_w, linewidth=0.8)
    bars[2].set_edgecolor("#F0C040")
    bars[2].set_linewidth(2.5)
    for j, (bar, v) in enumerate(zip(bars, mae_vals)):
        ax1.text(bar.get_x()+bar.get_width()/2, v+0.8, f"{v:.1f}",
                 ha='center', va='bottom', fontsize=11, fontweight='bold', color=colors[j])
    ax1.set_title("MAE Comparison  (lower = better)", fontsize=12, fontweight='bold', pad=8)
    ax1.set_ylabel("Mean Absolute Error (ms)", fontsize=11)
    ax1.set_ylim(0, max(mae_vals)*1.22)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    ax1.annotate("Best ML Model", xy=(2, mae_vals[2]),
                 xytext=(2.5, mae_vals[2]+10), fontsize=9,
                 color='#1E3A5F', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#1E3A5F', lw=1.2))

    # RMSE
    ax2 = fig.add_subplot(2, 2, 2)
    bars2 = ax2.bar(model_names, rmse_vals, color=colors, edgecolor=edge_colors, width=bar_w, linewidth=0.8)
    bars2[2].set_edgecolor("#F0C040")
    bars2[2].set_linewidth(2.5)
    for j, (bar, v) in enumerate(zip(bars2, rmse_vals)):
        ax2.text(bar.get_x()+bar.get_width()/2, v+1, f"{v:.1f}",
                 ha='center', va='bottom', fontsize=11, fontweight='bold', color=colors[j])
    ax2.set_title("RMSE Comparison  (lower = better)", fontsize=12, fontweight='bold', pad=8)
    ax2.set_ylabel("Root Mean Squared Error (ms)", fontsize=11)
    ax2.set_ylim(0, max(rmse_vals)*1.22)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    # Feature Importance
    ax3 = fig.add_subplot(2, 2, 3)
    feat_sorted = sorted(zip(FEATURE_COLS, gb.feature_importances_), key=lambda x: x[1])
    feat_names  = [f[0] for f in feat_sorted]
    feat_vals   = [f[1] for f in feat_sorted]
    bar_cols    = ["#BBBBBB","#BBBBBB","#8B4A1A","#0F4A3A","#1E3A5F","#1E3A5F"]
    hbars = ax3.barh(feat_names, feat_vals, color=bar_cols, edgecolor='white', height=0.55)
    for bar, v in zip(hbars, feat_vals):
        ax3.text(v+0.004, bar.get_y()+bar.get_height()/2, f"{v:.3f}",
                 va='center', fontsize=10, fontweight='bold')
    ax3.set_title("Feature Importance — Gradient Boosting", fontsize=12, fontweight='bold', pad=8)
    ax3.set_xlabel("Importance Score", fontsize=11)
    ax3.set_xlim(0, 0.78)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.xaxis.grid(True, alpha=0.3, linestyle='--')
    ax3.set_axisbelow(True)

    # Actual vs Predicted
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(y_test.values, gb_preds, color="#1E3A5F",
                alpha=0.8, edgecolors='white', s=70, zorder=3)
    mx = max(max(y_test.values), max(gb_preds)) * 1.08
    ax4.plot([0,mx],[0,mx], 'k--', linewidth=1.2, label='Perfect prediction', zorder=2)
    ax4.set_xlabel("Actual Execution Time (ms)", fontsize=11)
    ax4.set_ylabel("Predicted Execution Time (ms)", fontsize=11)
    ax4.set_title("Actual vs Predicted — Gradient Boosting", fontsize=12, fontweight='bold', pad=8)
    ax4.legend(fontsize=10)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax4.xaxis.grid(True, alpha=0.3, linestyle='--')
    ax4.set_axisbelow(True)

    legend_items = [
        mpatches.Patch(color=colors[0], label=f"PostgreSQL Planner   MAE={mae_vals[0]:.1f}ms"),
        mpatches.Patch(color=colors[1], label=f"Random Forest          MAE={mae_vals[1]:.1f}ms  ✓"),
        mpatches.Patch(color=colors[2], label=f"Gradient Boosting     MAE={mae_vals[2]:.1f}ms  ★ BEST"),
        mpatches.Patch(color=colors[3], label=f"SVR (RBF)                MAE={mae_vals[3]:.1f}ms"),
    ]
    fig.legend(handles=legend_items, loc='lower center', ncol=4,
               fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01),
               handlelength=1.2, handleheight=1.0)

    plt.tight_layout(rect=[0, 0.07, 1, 0.96])
    plt.savefig(CHART_PATH, dpi=180, bbox_inches="tight", facecolor='white')
    plt.show()
    print(f"\n  Chart saved to: {CHART_PATH}")


# ── Main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    X, y, X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, scaler = split_data(df)
    models, gb = train_models(X, y, X_train, X_test, y_train, y_test, X_train_sc, X_test_sc)
    print_results(models, y_test, X, y, gb)
    save_chart(models, y_test, gb)
    print("\n  ✅  Done!\n")
