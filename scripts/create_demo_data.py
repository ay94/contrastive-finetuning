"""
Creates demo data for the contrastive-experiment pipeline using the
20 Newsgroups dataset (sklearn). Outputs train.json, validation.json,
and exemplar.json in the expected JSONL format.

Usage:
    python scripts/create_demo_data.py --output ~/contrastive-experiments/contrastive-data

Categories used (4, for clear separation):
    - sci.space
    - rec.sport.hockey
    - talk.politics.guns
    - comp.graphics
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_20newsgroups


CATEGORIES = [
    "sci.space",
    "rec.sport.hockey",
    "talk.politics.guns",
    "comp.graphics",
]

LABEL_MAP = {
    "sci.space": "Space & Astronomy",
    "rec.sport.hockey": "Hockey & Ice Sports",
    "talk.politics.guns": "Gun Policy & Politics",
    "comp.graphics": "Computer Graphics",
}


def clean_text(text: str) -> str:
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()[:500]


def build_df(split: str, n_per_class: int) -> pd.DataFrame:
    data = fetch_20newsgroups(
        subset=split,
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
        random_state=42,
    )
    df = pd.DataFrame({"text": data.data, "target": data.target})
    df["class_label"] = df["target"].map(
        {i: LABEL_MAP[c] for i, c in enumerate(CATEGORIES)}
    )
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 40]
    df = (
        df.groupby("class_label", group_keys=False)
        .apply(lambda g: g.sample(min(n_per_class, len(g)), random_state=42))
        .reset_index(drop=True)
    )
    df["messageId"] = [f"msg_{i:05d}" for i in range(len(df))]
    df["topicId"] = df["class_label"]
    return df[["messageId", "class_label", "text", "topicId"]]


def save_jsonl(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved {len(df)} rows → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="~/contrastive-experiments/contrastive-data",
        help="Output directory for the three JSONL files",
    )
    parser.add_argument("--train-per-class", type=int, default=150)
    parser.add_argument("--val-per-class", type=int, default=50)
    parser.add_argument("--exemplar-per-class", type=int, default=20)
    args = parser.parse_args()

    out = Path(args.output).expanduser()
    print(f"Building demo data → {out}")
    print(f"Categories: {', '.join(LABEL_MAP.values())}\n")

    train_df = build_df("train", args.train_per_class)
    val_df = build_df("test", args.val_per_class)
    # Exemplars: small stratified sample from training set
    exemplar_df = (
        train_df.groupby("class_label")
        .sample(min(args.exemplar_per_class, args.train_per_class), random_state=0)
        .reset_index(drop=True)
    )
    exemplar_df["messageId"] = [f"ex_{i:04d}" for i in range(len(exemplar_df))]

    save_jsonl(train_df, out / "train.json")
    save_jsonl(val_df, out / "validation.json")
    save_jsonl(exemplar_df, out / "exemplar.json")

    print(f"\nClass distribution (train):")
    print(train_df["class_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
