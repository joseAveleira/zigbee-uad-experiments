"""Model evaluation for the Zigbee_UAD dataset.

The evaluation never uses `frame.number` or `frame.time_epoch` as model
inputs: both encode the position of a frame inside the recording session
and, because attacks were executed in timed bursts, they reveal the
labelling layout of the capture (data leakage).

Feature sets
------------
protocol  the 30 per-frame protocol/metadata fields
context   protocol + 6 causal traffic-context features (36)
reduced   subset of `context` selected by permutation importance

Stages (run in order; results accumulate in results/results.json):

    python3 run_experiments.py importance
    python3 run_experiments.py models
    python3 run_experiments.py svm --set protocol
    python3 run_experiments.py svm --set context
    python3 run_experiments.py svm --set reduced
    python3 run_experiments.py scenarios

The RBF-kernel SVM is fitted on a stratified subsample of the training
partition (--svm-cap, default 12000) because exact SVM training scales
quadratically with sample count; evaluation always uses the complete
test partition.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from preprocess import load_dataset

SEED = 42
IMPORTANCE_THRESHOLD = 0.001  # mean accuracy decrease under permutation


def get_split(df, out):
    split_file = out / "split.json"
    if split_file.exists():
        idx = json.loads(split_file.read_text())
        return np.array(idx["train"]), np.array(idx["test"])
    idx_train, idx_test = train_test_split(
        df.index, test_size=0.3, stratify=df["label"], random_state=SEED)
    split_file.write_text(json.dumps(
        {"train": idx_train.tolist(), "test": idx_test.tolist()}))
    return idx_train.to_numpy(), idx_test.to_numpy()


def load_results(out):
    f = out / "results.json"
    return json.loads(f.read_text()) if f.exists() else {}


def save_results(out, results):
    (out / "results.json").write_text(json.dumps(results, indent=2))


def scores(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_score)), 4),
    }


def rf(n=200, jobs=-1):
    return RandomForestClassifier(n_estimators=n, random_state=SEED,
                                  n_jobs=jobs)


def feature_sets(results, proto, ctx):
    return {"protocol": proto,
            "context": proto + ctx,
            "reduced": results.get("features_reduced", [])}


def stage_importance(df, proto, ctx, out):
    cols = proto + ctx
    X, y = df[cols], df["label"]
    idx_train, idx_test = get_split(df, out)
    model = rf(100)
    model.fit(X.loc[idx_train], y.loc[idx_train])
    model.set_params(n_jobs=1)
    ev, _ = train_test_split(idx_test, train_size=6000,
                             stratify=y.loc[idx_test], random_state=SEED)
    t0 = time.time()
    pi = permutation_importance(model, X.loc[ev], y.loc[ev],
                                n_repeats=5, random_state=SEED, n_jobs=1)
    print(f"permutation importance: {time.time() - t0:.0f}s")
    imp = pd.DataFrame({"feature": cols, "mean": pi.importances_mean,
                        "std": pi.importances_std}
                       ).sort_values("mean", ascending=False)
    imp.to_csv(out / "importances.csv", index=False)
    reduced = imp.loc[imp["mean"] > IMPORTANCE_THRESHOLD, "feature"].tolist()
    results = load_results(out)
    results.update({
        "n_frames": len(df),
        "n_features_protocol": len(proto),
        "n_features_context": len(cols),
        "features_protocol": proto, "features_context_extra": ctx,
        "split": "70/30 stratified", "seed": SEED,
        "importance_threshold": IMPORTANCE_THRESHOLD,
        "features_reduced": reduced, "n_features_reduced": len(reduced),
    })
    save_results(out, results)
    print(f"reduced subset ({len(reduced)}):")
    print(imp.head(len(reduced) + 4).to_string(index=False))


def stage_models(df, proto, ctx, out):
    results = load_results(out)
    sets = feature_sets(results, proto, ctx)
    idx_train, idx_test = get_split(df, out)
    models = {
        "Random Forest": rf(),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=6,
                                 learning_rate=0.1, eval_metric="logloss",
                                 random_state=SEED, n_jobs=-1),
        "k-NN": make_pipeline(StandardScaler(),
                              KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=SEED)),
    }
    for tag, cols in sets.items():
        results.setdefault(tag, {})
        for name, model in models.items():
            t0 = time.time()
            model.fit(df.loc[idx_train, cols], df.loc[idx_train, "label"])
            res = scores(model, df.loc[idx_test, cols],
                         df.loc[idx_test, "label"])
            res["train_seconds"] = round(time.time() - t0, 1)
            results[tag][name] = res
            print(tag, name, res)
    save_results(out, results)


def stage_svm(df, proto, ctx, out, cap, which):
    results = load_results(out)
    sets = feature_sets(results, proto, ctx)
    cols = sets[which]
    idx_train, idx_test = get_split(df, out)
    y = df["label"]
    if cap and len(idx_train) > cap:
        sub, _ = train_test_split(idx_train, train_size=cap,
                                  stratify=y.loc[idx_train],
                                  random_state=SEED)
        results["svm_train_frames"] = int(cap)
    else:
        sub = idx_train
        results["svm_train_frames"] = int(len(idx_train))
    model = make_pipeline(StandardScaler(),
                          SVC(kernel="rbf", C=1.0, gamma="scale",
                              random_state=SEED))
    t0 = time.time()
    model.fit(df.loc[sub, cols], y.loc[sub])
    res = scores(model, df.loc[idx_test, cols], y.loc[idx_test])
    res["train_seconds"] = round(time.time() - t0, 1)
    results.setdefault(which, {})["SVM (RBF)"] = res
    print(which, "SVM (RBF)", res)
    save_results(out, results)


def stage_leakage(df, proto, ctx, out):
    """Quantify the label leakage caused by capture-position fields.

    Trains the four fast models on protocol attributes PLUS the two
    excluded position fields (frame.number, frame.time_epoch), and a
    Random Forest on the two position fields ALONE.  These results are
    reported in the paper only to demonstrate the leakage; the fields
    are never part of any legitimate feature set.
    """
    from preprocess import LEAKY_COLS
    results = load_results(out)
    idx_train, idx_test = get_split(df, out)
    y = df["label"]
    models = {
        "Random Forest": rf(),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=6,
                                 learning_rate=0.1, eval_metric="logloss",
                                 random_state=SEED, n_jobs=-1),
        "k-NN": make_pipeline(StandardScaler(),
                              KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=SEED)),
    }
    cols = LEAKY_COLS + proto
    results["leakage"] = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(df.loc[idx_train, cols], y.loc[idx_train])
        res = scores(model, df.loc[idx_test, cols], y.loc[idx_test])
        res["train_seconds"] = round(time.time() - t0, 1)
        results["leakage"][name] = res
        print("leakage", name, res)
    m = rf()
    m.fit(df.loc[idx_train, LEAKY_COLS], y.loc[idx_train])
    results["leakage_position_only_rf"] = scores(
        m, df.loc[idx_test, LEAKY_COLS], y.loc[idx_test])
    print("position-only RF", results["leakage_position_only_rf"])
    save_results(out, results)


def stage_scenarios(df, proto, ctx, out):
    results = load_results(out)
    reduced = results["features_reduced"]
    idx_train, idx_test = get_split(df, out)
    model = rf()
    model.fit(df.loc[idx_train, reduced], df.loc[idx_train, "label"])
    results["per_scenario_rf_reduced"] = {}
    for scen, part in df.loc[idx_test].groupby("scenario"):
        results["per_scenario_rf_reduced"][scen] = scores(
            model, part[reduced], part["label"])
        print("scenario", scen, results["per_scenario_rf_reduced"][scen])
    results["loso_rf_reduced"] = {}
    for scen in df["scenario"].unique():
        tr, te = df[df["scenario"] != scen], df[df["scenario"] == scen]
        m = rf()
        m.fit(tr[reduced], tr["label"])
        results["loso_rf_reduced"][scen] = scores(m, te[reduced], te["label"])
        print("LOSO", scen, results["loso_rf_reduced"][scen])
    save_results(out, results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["importance", "models", "svm",
                                      "scenarios", "leakage"])
    ap.add_argument("--data", default="../../../recursos/datasets_zigbee")
    ap.add_argument("--out", default="results")
    ap.add_argument("--svm-cap", type=int, default=12000)
    ap.add_argument("--set", default="context",
                    choices=["protocol", "context", "reduced"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    df, proto, ctx = load_dataset(args.data)
    {"importance": lambda: stage_importance(df, proto, ctx, out),
     "models": lambda: stage_models(df, proto, ctx, out),
     "svm": lambda: stage_svm(df, proto, ctx, out, args.svm_cap, args.set),
     "scenarios": lambda: stage_scenarios(df, proto, ctx, out),
     "leakage": lambda: stage_leakage(df, proto, ctx, out),
     }[args.stage]()
