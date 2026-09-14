
import argparse
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
)

import shap

warnings.filterwarnings("ignore", category=UserWarning)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# --------------------------------------------------------------------------
# 1. Feature selection: Mann-Whitney U, fit within a training fold ONLY
# --------------------------------------------------------------------------
def mannwhitney_select(X_train, y_train, alpha=0.05):
    """Return the list of columns with a significant PD-vs-HC difference
    (two-sided Mann-Whitney U test, p < alpha) computed on X_train only."""
    selected = []
    class0 = X_train[y_train == 0]
    class1 = X_train[y_train == 1]
    for col in X_train.columns:
        a, b = class0[col].values, class1[col].values
        if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
            continue  # constant, identical column -> undefined test
        try:
            _, p = mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            continue
        if p < alpha:
            selected.append(col)
    return selected


# --------------------------------------------------------------------------
# 2. Hyperparameter search space (Random Forest block of Table 7)
# --------------------------------------------------------------------------
RF_PARAM_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 10, 20, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False],
}


# --------------------------------------------------------------------------
# 3. Extract positive-class SHAP values/base-value across shap versions
# --------------------------------------------------------------------------
def positive_class_shap(explanation, pos_index=1):
    """Handle both 2D (n, features) and 3D (n, features, classes) outputs
    across shap versions and return (values, base_values) for the PD class."""
    values = explanation.values
    base = explanation.base_values
    if values.ndim == 3:
        values = values[:, :, pos_index]
        base = base[:, pos_index] if np.ndim(base) > 1 else base
    return values, base


# --------------------------------------------------------------------------
# 4. Main nested-CV + out-of-fold SHAP pipeline
# --------------------------------------------------------------------------
def run_nested_cv_shap(
    df, feature_cols, label_col, group_col, pos_label=1,
    n_outer=10, n_inner=5, n_iter_search=30, alpha=0.05, outdir="shap_out",
    checkpoint_path=None, time_budget_seconds=None,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X_full = df[feature_cols].reset_index(drop=True)
    y = (df[label_col].values == pos_label).astype(int)

    group_raw = df[group_col].astype(str)
    if group_raw.nunique() == len(df):
        # group_col is unique per row (e.g. segment IDs like "PD1_1",
        # "PD1_2", ...) rather than one value per participant. Derive a
        # participant-level grouping by stripping the trailing
        # "_<segment number>" suffix, so the group-wise split still keeps
        # each participant's segments entirely in train OR test.
        derived = group_raw.str.rsplit("_", n=1).str[0]
        if derived.nunique() < len(df):
            print(f"Group column '{group_col}' has a unique value per row "
                  f"({len(df)} segments) -- treating it as a segment ID and "
                  f"deriving participant-level groups by stripping the "
                  f"trailing '_<segment>' suffix: "
                  f"{derived.nunique()} unique participants found.")
            groups = derived.values
        else:
            print(f"WARNING: could not derive a coarser grouping from "
                  f"'{group_col}'; every value is unique and no common "
                  f"prefix was found. Proceeding with '{group_col}' as-is, "
                  f"which means group-wise CV has no effect (equivalent to "
                  f"a plain stratified split). Check --group-col.")
            groups = group_raw.values
    else:
        groups = group_raw.values

    outer_cv = StratifiedGroupKFold(
        n_splits=n_outer, shuffle=True, random_state=RANDOM_SEED
    )
    outer_splits = list(outer_cv.split(X_full, y, groups))

    # ---- checkpoint support -----------------------------------------------
    # Lets a full run (e.g. n_outer=10, n_inner=5, n_iter_search=30, which
    # can mean ~1500 RF fits) be completed across several shorter script
    # invocations without redoing finished folds. Safe to ignore entirely
    # if you run this in an environment with no wall-clock limit.
    ckpt_path = Path(checkpoint_path) if checkpoint_path else None
    done = {}
    if ckpt_path is not None and ckpt_path.exists():
        with open(ckpt_path, "rb") as fh:
            done = pickle.load(fh)
        print(f"Resuming from checkpoint '{ckpt_path}': "
              f"{len(done)}/{n_outer} folds already completed.\n")

    start_time = time.time()

    print(f"Running {n_outer}-fold group-wise nested CV "
          f"({len(df)} samples, {len(np.unique(groups))} speakers/groups)...\n")

    for fold_idx, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        if fold_idx in done:
            continue  # already computed in a previous invocation

        if (time_budget_seconds is not None
                and time.time() - start_time > time_budget_seconds):
            print(f"\nTime budget of {time_budget_seconds}s reached after "
                  f"{len(done)}/{n_outer} folds. Progress has been saved to "
                  f"'{ckpt_path}'. Re-run the exact same command to "
                  f"continue from where it left off.")
            break

        X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        # ---- feature selection: TRAIN FOLD ONLY --------------------------
        selected = mannwhitney_select(X_train, y_train, alpha=alpha)
        X_train_sel = X_train[selected]
        X_test_sel = X_test[selected]

        # ---- inner k-fold hyperparameter tuning ---------------------------
        inner_cv = StratifiedGroupKFold(
            n_splits=n_inner, shuffle=True, random_state=RANDOM_SEED
        )
        search = RandomizedSearchCV(
            RandomForestClassifier(random_state=RANDOM_SEED),
            param_distributions=RF_PARAM_GRID,
            n_iter=n_iter_search,
            cv=inner_cv,
            scoring="accuracy",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        search.fit(X_train_sel, y_train, groups=groups_train)
        best_rf = search.best_estimator_

        # ---- held-out evaluation ------------------------------------------
        proba = best_rf.predict_proba(X_test_sel)[:, 1]
        pred = (proba >= 0.5).astype(int)

        fold_metric = {
            "fold": fold_idx,
            "n_features_selected": len(selected),
            "n_test": len(test_idx),
            "accuracy": accuracy_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "auc": roc_auc_score(y_test, proba),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
        }

        # ---- SHAP on the held-out test fold ONLY ---------------------------
        explainer = shap.TreeExplainer(best_rf)
        explanation = explainer(X_test_sel)
        values, base_values = positive_class_shap(explanation, pos_index=1)

        done[fold_idx] = {
            "fold": fold_idx,
            "test_index": X_test.index,
            "selected": selected,
            "shap_values": values,
            "base_value": float(np.mean(base_values)),
            "X_test_sel": X_test_sel,
            "y_test": y_test,
            "proba": proba,
            "pred": pred,
            "metrics": fold_metric,
        }

        if ckpt_path is not None:
            tmp_path = ckpt_path.with_suffix(".tmp")
            with open(tmp_path, "wb") as fh:
                pickle.dump(done, fh)
            tmp_path.replace(ckpt_path)  # atomic on POSIX

        print(f"  fold {fold_idx:2d}/{n_outer}: "
              f"{len(selected)} features selected | "
              f"acc={fold_metric['accuracy']:.3f} "
              f"f1={fold_metric['f1']:.3f} "
              f"auc={fold_metric['auc']:.3f}  "
              f"[{time.time() - start_time:.0f}s elapsed]")

    if len(done) < n_outer:
        print(f"\n{len(done)}/{n_outer} folds completed so far. "
              f"Re-run the same command to continue -- already-finished "
              f"folds are skipped automatically.")
        return None  # signal "not finished yet" to caller

    # ---- assemble full results from (possibly checkpoint-restored) folds --
    fold_records = [done[i] for i in range(1, n_outer + 1)]
    fold_selected_features = [r["selected"] for r in fold_records]
    fold_feature_ranks = []
    pooled_shap = pd.DataFrame(np.nan, index=X_full.index, columns=feature_cols)
    oof_pred = np.full(len(df), np.nan)
    oof_proba = np.full(len(df), np.nan)
    fold_metrics = []

    for r in fold_records:
        pooled_shap.loc[r["test_index"], r["selected"]] = r["shap_values"]
        mean_abs = pd.Series(
            np.abs(r["shap_values"]).mean(axis=0), index=r["selected"]
        ).sort_values(ascending=False)
        fold_feature_ranks.append(
            pd.Series(np.arange(1, len(mean_abs) + 1), index=mean_abs.index)
        )
        oof_pred[r["test_index"]] = r["pred"]
        oof_proba[r["test_index"]] = r["proba"]
        fold_metrics.append(r["metrics"])

    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv(outdir / "fold_level_performance.csv", index=False)
    print("\nAll folds complete. Mean +/- SD across the 10 outer folds "
          "(should reproduce Table 8's RF + Mann-Whitney U row):")
    print(metrics_df[["accuracy", "f1", "auc", "precision", "recall"]]
          .agg(["mean", "std"]).round(4))

    return {
        "pooled_shap": pooled_shap,
        "fold_selected_features": fold_selected_features,
        "fold_feature_ranks": fold_feature_ranks,
        "fold_records": fold_records,
        "oof_pred": oof_pred,
        "oof_proba": oof_proba,
        "metrics_df": metrics_df,
        "X_full": X_full,
        "y": y,
        "outdir": outdir,
    }


# --------------------------------------------------------------------------
# 5. Figure 4: pooled out-of-fold SHAP beeswarm (intersection of features
#    selected in ALL 10 folds, to keep the pooled matrix fully populated
#    without fabricating any values)
# --------------------------------------------------------------------------
def make_figure4(results, max_display=20):
    pooled_shap = results["pooled_shap"]
    X_full = results["X_full"]
    fold_selected_features = results["fold_selected_features"]
    outdir = results["outdir"]

    common_features = set(fold_selected_features[0])
    for feats in fold_selected_features[1:]:
        common_features &= set(feats)
    common_features = sorted(common_features)

    print(f"\nFigure 4: {len(common_features)} features were selected in "
          f"ALL 10 outer folds and are used for the pooled beeswarm plot.")

    shap_matrix = pooled_shap[common_features].values
    feature_matrix = X_full[common_features].values

    explanation = shap.Explanation(
        values=shap_matrix,
        data=feature_matrix,
        feature_names=common_features,
    )

    plt.figure()
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    plt.title("SHAP summary (out-of-fold, pooled across 10 outer folds)")
    plt.tight_layout()
    fig_path = outdir / "figure4_shap_summary_pooled_oof.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {fig_path}")
    return common_features


# --------------------------------------------------------------------------
# 6. Figures 5 & 6: waterfall plots for one representative PD and one
#    representative HC instance, each explained with its own fold's model
# --------------------------------------------------------------------------
def make_figures_5_6(results, max_display=10):
    fold_records = results["fold_records"]
    oof_proba = results["oof_proba"]
    oof_pred = results["oof_pred"]
    y = results["y"]
    outdir = results["outdir"]

    correct_pd = np.where((oof_pred == 1) & (y == 1))[0]
    correct_hc = np.where((oof_pred == 0) & (y == 0))[0]
    pd_instance_global_idx = correct_pd[np.argmax(oof_proba[correct_pd])]
    hc_instance_global_idx = correct_hc[np.argmin(oof_proba[correct_hc])]

    def render_waterfall(global_idx, title, fname):
        rec = next(
            r for r in fold_records if global_idx in r["test_index"]
        )
        local_pos = list(rec["test_index"]).index(global_idx)
        exp = shap.Explanation(
            values=rec["shap_values"][local_pos],
            base_values=rec["base_value"],
            data=rec["X_test_sel"].iloc[local_pos].values,
            feature_names=rec["selected"],
        )
        plt.figure()
        shap.plots.waterfall(exp, max_display=max_display, show=False)
        plt.title(title)
        plt.tight_layout()
        fig_path = outdir / fname
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  saved -> {fig_path}  "
              f"(fold {rec['fold']}, predicted P(PD)="
              f"{rec['proba'][local_pos]:.3f})")

    print("\nFigure 5 (representative PD instance):")
    render_waterfall(
        pd_instance_global_idx,
        "SHAP waterfall — representative PD instance (out-of-fold)",
        "figure5_shap_waterfall_PD.png",
    )

    print("Figure 6 (representative HC instance):")
    render_waterfall(
        hc_instance_global_idx,
        "SHAP waterfall — representative HC instance (out-of-fold)",
        "figure6_shap_waterfall_HC.png",
    )


# --------------------------------------------------------------------------
# 7. Supplementary table: feature-rank stability across the 10 outer folds
#    (directly parallels the fold-range style already used in Table 6)
# --------------------------------------------------------------------------
def make_rank_stability_table(results, top_n=20):
    fold_feature_ranks = results["fold_feature_ranks"]
    n_folds = len(fold_feature_ranks)
    outdir = results["outdir"]

    rank_df = pd.concat(fold_feature_ranks, axis=1)
    rank_df.columns = [f"fold_{i+1}" for i in range(n_folds)]

    summary = pd.DataFrame({
        "n_folds_selected": rank_df.notna().sum(axis=1),
        "mean_rank": rank_df.mean(axis=1, skipna=True),
        "min_rank": rank_df.min(axis=1, skipna=True),
        "max_rank": rank_df.max(axis=1, skipna=True),
    })
    summary["rank_range"] = (
        summary["min_rank"].astype(int).astype(str) + "-"
        + summary["max_rank"].astype(int).astype(str)
    )
    summary = summary.sort_values(
        ["n_folds_selected", "mean_rank"], ascending=[False, True]
    )

    out_path = outdir / "supplementary_feature_rank_stability.csv"
    summary.to_csv(out_path)
    print(f"\nSupplementary rank-stability table saved -> {out_path}")
    print(summary.head(top_n).round(2).to_string())
    return summary


# --------------------------------------------------------------------------
# 8. Synthetic demo data (ONLY for pipeline verification — replace with
#    --data pointing at your real feature table for the real analysis)
# --------------------------------------------------------------------------
def make_synthetic_demo(n_participants=120, segs_per_participant=(5, 8),
                         n_features=71, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    rows = []
    feature_names = [f"feat_{i:02d}" for i in range(n_features)]
    for pid in range(n_participants):
        label = 1 if pid < n_participants // 2 else 0  # 1=PD, 0=HC
        n_seg = rng.integers(segs_per_participant[0], segs_per_participant[1] + 1)
        speaker_shift = rng.normal(0, 0.3, size=n_features)
        for _ in range(n_seg):
            base = rng.normal(0, 1, size=n_features) + speaker_shift
            if label == 1:
                base[:15] += rng.normal(0.6, 0.2, size=15)  # informative feats
            rows.append([pid, label] + base.tolist())
    cols = ["participant_id", "label"] + feature_names
    return pd.DataFrame(rows, columns=cols), feature_names


# --------------------------------------------------------------------------
# 9. CLI
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str,
                         default="/mnt/user-data/uploads/BenSParX__1_.csv",
                         help="Path to CSV with features + label + group "
                              "columns. Defaults to the uploaded BenSParX "
                              "dataset. In Colab, override with --data "
                              "/content/BenSParX.csv. If the file is not "
                              "found, the script falls back to the "
                              "synthetic demo so the pipeline can still be "
                              "smoke-tested.")
    parser.add_argument("--label-col", type=str, default="class")
    parser.add_argument("--group-col", type=str, default="id")
    parser.add_argument("--pos-label", type=int, default=1,
                         help="Value in label-col that means PD.")
    parser.add_argument("--outdir", type=str, default="shap_out")
    parser.add_argument("--n-outer", type=int, default=10)
    parser.add_argument("--n-inner", type=int, default=5)
    parser.add_argument("--n-iter-search", type=int, default=30,
                         help="RandomizedSearchCV iterations per outer fold.")
    parser.add_argument("--alpha", type=float, default=0.05,
                         help="Mann-Whitney U significance threshold.")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pkl",
                         help="Path to a checkpoint file. Progress is saved "
                              "after every completed outer fold; re-running "
                              "the same command resumes automatically "
                              "instead of redoing finished folds. Delete "
                              "this file to force a clean restart.")
    parser.add_argument("--time-budget-seconds", type=int, default=None,
                         help="Stop after this many seconds (saving "
                              "checkpoint progress) even if not all folds "
                              "are done -- useful when running in short "
                              "chunks. Omit for no limit.")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"'{args.data}' not found in the current working directory.\n"
              f"  -> Place your real feature table there (or pass "
              f"--data /path/to/your_file.csv) once it's ready.\n"
              f"  -> Falling back to synthetic demo data for now "
              f"(FOR PIPELINE VERIFICATION ONLY -- not real results).\n")
        df, feature_cols = make_synthetic_demo()
        label_col, group_col = args.label_col, args.group_col
    else:
        df = pd.read_csv(data_path)
        label_col, group_col = args.label_col, args.group_col

        # be forgiving about column-name casing/variants so this doesn't
        # break on the first run just because of naming differences
        def _resolve_col(requested, df_cols, candidates):
            if requested in df_cols:
                return requested
            lower_map = {c.lower(): c for c in df_cols}
            if requested.lower() in lower_map:
                return lower_map[requested.lower()]
            for cand in candidates:
                if cand in lower_map:
                    return lower_map[cand]
            raise ValueError(
                f"Could not find a '{requested}' column in {data_path}. "
                f"Available columns: {list(df_cols)}. "
                f"Pass --label-col / --group-col explicitly."
            )

        label_col = _resolve_col(
            label_col, df.columns,
            ["label", "class", "target", "diagnosis", "group"]
        )
        group_col = _resolve_col(
            group_col, df.columns,
            ["participant_id", "speaker_id", "subject_id", "patient_id", "id"]
        )

        # allow string labels like "PD"/"HC" in addition to 0/1
        if df[label_col].dtype == object:
            uniq = sorted(df[label_col].dropna().unique().tolist())
            print(f"Label column '{label_col}' has string values {uniq}; "
                  f"mapping to 0/1 (assuming PD-like value -> 1).")
            pd_like = [v for v in uniq if str(v).strip().lower() in
                       ("pd", "parkinson", "parkinsons", "1", "positive")]
            pos_val = pd_like[0] if pd_like else uniq[-1]
            df[label_col] = (df[label_col] == pos_val).astype(int)
            args.pos_label = 1

        feature_cols = [
            c for c in df.columns
            if c not in (label_col, group_col)
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        print(f"Loaded {data_path}: {len(df)} rows, "
              f"{len(feature_cols)} candidate numeric feature columns "
              f"(label_col='{label_col}', group_col='{group_col}').\n")

    results = run_nested_cv_shap(
        df, feature_cols, label_col, group_col,
        pos_label=args.pos_label, n_outer=args.n_outer, n_inner=args.n_inner,
        n_iter_search=args.n_iter_search, alpha=args.alpha, outdir=args.outdir,
        checkpoint_path=args.checkpoint,
        time_budget_seconds=args.time_budget_seconds,
    )
    if results is None:
        return  # not all folds finished yet; re-run to continue
    make_figure4(results)
    make_figures_5_6(results)
    make_rank_stability_table(results)
    print("\nDone. All figures and tables written to:", results["outdir"])


if __name__ == "__main__":
    main()
