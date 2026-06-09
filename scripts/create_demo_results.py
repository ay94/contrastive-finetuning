#!/usr/bin/env python3
"""
Creates realistic dummy results for the contrastive-experiment dashboard.

Generates fake-but-plausible outputs for a baseline model and a fine-tuned
checkpoint so the Streamlit dashboard can be launched and screenshotted without
running the actual training pipeline.

Usage:
    python scripts/create_demo_results.py
    python scripts/create_demo_results.py --base-dir ~/my-experiments
    streamlit run dashboard/Load.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = [
    "Space & Astronomy",
    "Hockey & Ice Sports",
    "Gun Policy & Politics",
    "Computer Graphics",
]

MODEL_DIR_NAME = "all_minilm_l6_v2"

# 2D UMAP cluster centres per class
CENTRES = np.array([
    [6.0,  3.0],   # Space & Astronomy
    [-4.0, 4.0],   # Hockey & Ice Sports
    [-3.0, -5.0],  # Gun Policy & Politics
    [5.0,  -4.0],  # Computer Graphics
])

MODELS = [
    {
        "name": "Baseline",
        "config": None,
        "std": 1.5,
        "sil_mean": 0.35,
        "f1_base": 0.74,
        "sim_diag": 0.52,
        "sim_off": 0.23,
        "avg_sim": 0.49,
    },
    {
        "name": "checkpoint-31",
        "config": "contrastive_fixed",
        "std": 0.75,
        "sil_mean": 0.54,
        "f1_base": 0.84,
        "sim_diag": 0.73,
        "sim_off": 0.14,
        "avg_sim": 0.71,
    },
]

N_TRAIN = 150
N_TEST = 50


def _rng(seed_str):
    return np.random.default_rng(abs(hash(seed_str)) % 2**32)


def make_semantic_map(m, split):
    n = N_TRAIN if split == "train" else N_TEST
    rng = _rng(m["name"] + split + "umap")
    rows = []
    for i, cls in enumerate(CLASSES):
        coords = rng.normal(loc=CENTRES[i], scale=m["std"], size=(n, 2))
        sils = rng.normal(loc=m["sil_mean"], scale=0.08, size=n).clip(-1, 1)
        for j, (xy, sil) in enumerate(zip(coords, sils)):
            rows.append({
                "messageId": f"{split}_{i * n + j:05d}",
                "class_label": cls,
                "x": round(float(xy[0]), 4),
                "y": round(float(xy[1]), 4),
                "true_score": round(float(sil), 4),
                "Split": split,
            })
    return pd.DataFrame(rows)


def make_report(m, split="test"):
    rng = _rng(m["name"] + split + "report")
    rows = []
    for cls in CLASSES:
        p = rng.uniform(m["f1_base"] - 0.04, m["f1_base"] + 0.06)
        r = rng.uniform(m["f1_base"] - 0.04, m["f1_base"] + 0.06)
        f1 = 2 * p * r / (p + r)
        rows.append({
            "class": cls,
            "precision": round(float(p), 3),
            "recall": round(float(r), 3),
            "f1-score": round(float(f1), 3),
            "support": float(N_TEST),
            "Split": split,
        })
    macro_p = np.mean([r["precision"] for r in rows])
    macro_r = np.mean([r["recall"] for r in rows])
    macro_f1 = np.mean([r["f1-score"] for r in rows])
    total = float(N_TEST * len(CLASSES))
    rows += [
        {"class": "accuracy",     "precision": round(macro_f1, 3), "recall": round(macro_f1, 3), "f1-score": round(macro_f1, 3), "support": total, "Split": split},
        {"class": "macro avg",    "precision": round(macro_p, 3),  "recall": round(macro_r, 3),  "f1-score": round(macro_f1, 3), "support": total, "Split": split},
        {"class": "weighted avg", "precision": round(macro_p, 3),  "recall": round(macro_r, 3),  "f1-score": round(macro_f1, 3), "support": total, "Split": split},
    ]
    return pd.DataFrame(rows)


def make_avg_similarities(m, split):
    rng = _rng(m["name"] + split + "avgsim")
    rows = [
        {
            "Class Label": cls,
            "Average Similarity": round(float(rng.uniform(m["avg_sim"] - 0.04, m["avg_sim"] + 0.04)), 4),
            "Split": split,
        }
        for cls in CLASSES
    ]
    return pd.DataFrame(rows)


def make_similarity_matrix(m, split):
    rng = _rng(m["name"] + split + "simmat")
    n = len(CLASSES)
    mat = np.full((n, n), m["sim_off"])
    np.fill_diagonal(mat, m["sim_diag"])
    noise = rng.normal(0, 0.015, size=(n, n))
    mat = np.clip(mat + noise, 0, 1)
    mat = (mat + mat.T) / 2  # symmetric
    np.fill_diagonal(mat, m["sim_diag"] + rng.uniform(-0.01, 0.01))
    df = pd.DataFrame(mat.round(4), index=CLASSES, columns=CLASSES).reset_index()
    df = df.rename(columns={"index": "index"})
    df["Split"] = split
    return df


def make_annotated(m):
    rng = _rng(m["name"] + "annotated")
    accuracy = m["f1_base"] - 0.02
    rows = []
    for i, cls in enumerate(CLASSES):
        for j in range(N_TEST):
            correct = rng.random() < accuracy
            pred = cls if correct else rng.choice([c for c in CLASSES if c != cls])
            sil = rng.normal(loc=m["avg_sim"] - 0.05, scale=0.08)
            rows.append({
                "messageId": f"val_{i * N_TEST + j:05d}",
                "class_label": cls,
                "predicted_label": pred,
                "pred_score": round(float(np.clip(sil, -1, 1)), 4),
            })
    return pd.DataFrame(rows)


def make_combined(report, sem_test, ann, avg_sim_test, m, experiment, model_name):
    macro = report[report["class"] == "macro avg"].iloc[0]
    weighted = report[report["class"] == "weighted avg"].iloc[0]
    return pd.DataFrame([{
        "macro_precision":    round(float(macro["precision"]), 4),
        "macro_recall":       round(float(macro["recall"]), 4),
        "macro_f1-score":     round(float(macro["f1-score"]), 4),
        "weighted_precision": round(float(weighted["precision"]), 4),
        "weighted_recall":    round(float(weighted["recall"]), 4),
        "weighted_f1-score":  round(float(weighted["f1-score"]), 4),
        "true_silhouette":    round(float(sem_test["true_score"].mean()), 4),
        "pred_silhouette":    round(float(ann["pred_score"].mean()), 4),
        "average_similarity": round(float(avg_sim_test["Average Similarity"].mean()), 4),
        "support":            float(N_TEST * len(CLASSES)),
        "Config":             m["config"],
        "Experiment":         experiment,
        "Model Name":         model_name,
        "Model":              m["name"],
    }])


def generate(base_folder: Path):
    model_dir = base_folder / "contrastive-experiment" / MODEL_DIR_NAME
    out_dir = model_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    # experiment = model_dir.parts[-3] matches Results.read_report() logic
    experiment = model_dir.parts[-3]

    buckets = {k: [] for k in [
        "reports", "semantic_maps", "average_similarities",
        "similarity_matrices", "annotated_datasets",
        "unmasked_annotated_datasets", "combined_metrics",
    ]}

    for m in MODELS:
        print(f"Generating: {m['name']}")
        meta = {"Experiment": experiment, "Model Name": MODEL_DIR_NAME, "Model": m["name"], "Config": m["config"]}

        for split in ["train", "test"]:
            buckets["semantic_maps"].append(make_semantic_map(m, split).assign(**meta))
            buckets["average_similarities"].append(make_avg_similarities(m, split).assign(**meta))
            buckets["similarity_matrices"].append(make_similarity_matrix(m, split).assign(**meta))

        report = make_report(m).assign(**meta)
        buckets["reports"].append(report)

        ann = make_annotated(m).assign(**meta)
        buckets["annotated_datasets"].append(ann)
        buckets["unmasked_annotated_datasets"].append(ann.copy())

        sem_test = make_semantic_map(m, "test")
        avg_sim_test = make_avg_similarities(m, "test")
        combined = make_combined(report, sem_test, ann, avg_sim_test, m, experiment, MODEL_DIR_NAME)
        buckets["combined_metrics"].append(combined)

    for name, dfs in buckets.items():
        df = pd.concat(dfs, ignore_index=True)
        path = out_dir / f"{name}.json"
        df.to_json(path, orient="records", lines=True)
        print(f"  {name}.json → {len(df)} rows")

    print(f"\nDone. Launch: streamlit run dashboard/Load.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=None)
    args = parser.parse_args()

    # Resolve base folder without importing contrastive_experiment (no ML deps needed)
    if args.base_dir:
        base = Path(args.base_dir).expanduser()
    else:
        import os
        env = os.environ.get("CONTRASTIVE_BASE_FOLDER")
        base = Path(env).expanduser() if env else Path("~/contrastive-experiments").expanduser()

    print(f"Base folder: {base}")
    generate(base)


if __name__ == "__main__":
    main()
