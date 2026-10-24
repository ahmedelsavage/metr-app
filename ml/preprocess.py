import json
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from ml.config import LABEL_NAMES_FILE, PROCESSED_DIR, RAW_DIR, REPORT_FILE, SPLITS


def parse_labels(value):
    return [int(x) for x in re.findall(r"\d+", str(value))]


def normalize_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def load_raw():
    frames = {}
    for name in SPLITS:
        df = pd.read_csv(RAW_DIR / f"{name}.csv")
        df["text"] = df["text"].apply(normalize_text)
        df["labels"] = df["labels"].apply(parse_labels)
        df["lab_key"] = df["labels"].apply(lambda x: tuple(sorted(x)))
        frames[name] = df
    return frames


def clean_train(train, val, test):
    val_texts, test_texts = set(val["text"]), set(test["text"])
    report = {
        "train_rows_before": len(train),
        "dups_inside": {
            "train": int(train["text"].duplicated().sum()),
            "val": int(val["text"].duplicated().sum()),
            "test": int(test["text"].duplicated().sum()),
        },
        "train_val_overlap": int(train["text"].isin(val_texts).sum()),
        "train_test_overlap": int(train["text"].isin(test_texts).sum()),
        "val_test_overlap": int(val["text"].isin(test_texts).sum()),
        "conflicting_texts": int(
            (train.groupby("text")["lab_key"].nunique() > 1).sum()
        ),
    }

    n = len(train)
    train = train[train["text"] != ""]
    report["empty_text_removed"] = n - len(train)

    n = len(train)
    train = train.drop_duplicates(subset=["text", "lab_key"])
    report["exact_duplicates_removed"] = n - len(train)

    n = len(train)
    train = train[~train["text"].isin(val_texts | test_texts)]
    report["leakage_rows_removed"] = n - len(train)

    n = len(train)
    conflicts = train.groupby("text")["lab_key"].nunique()
    train = train[~train["text"].isin(conflicts[conflicts > 1].index)]
    report["conflict_rows_removed"] = n - len(train)

    train = train.reset_index(drop=True)
    report["train_rows_after"] = len(train)
    return train, report


def binarize(frames, n_classes):
    mlb = MultiLabelBinarizer(classes=list(range(n_classes)))
    mlb.fit(frames["train"]["labels"])
    return {name: mlb.transform(df["labels"]) for name, df in frames.items()}


def build_output(df, Y, label_names):
    out = pd.DataFrame(Y, columns=label_names)
    out.insert(0, "text", df["text"].values)
    out["is_unfair"] = (Y.sum(axis=1) > 0).astype(int)
    return out


def validate(frames, targets, label_names):
    for name, df in frames.items():
        assert len(df) == targets[name].shape[0], f"{name}: text/label length mismatch"
        assert targets[name].shape[1] == len(label_names), f"{name}: wrong label count"
        assert (df["text"] != "").all(), f"{name}: empty text"
    train_texts = set(frames["train"]["text"])
    assert not train_texts & set(frames["val"]["text"]), "train-val leakage"
    assert not train_texts & set(frames["test"]["text"]), "train-test leakage"
    assert not (frames["train"].groupby("text")["lab_key"].nunique() > 1).any(), (
        "conflicts remain"
    )


def save(frames, targets, label_names):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        build_output(df, targets[name], label_names).to_csv(
            PROCESSED_DIR / f"{name}_clean.csv", index=False
        )


def main():
    label_names = json.loads(LABEL_NAMES_FILE.read_text())
    frames = load_raw()
    frames["train"], report = clean_train(
        frames["train"], frames["val"], frames["test"]
    )
    targets = binarize(frames, len(label_names))
    validate(frames, targets, label_names)
    save(frames, targets, label_names)

    report["rows_final"] = {name: len(df) for name, df in frames.items()}
    report["label_counts"] = {
        name: dict(zip(label_names, map(int, Y.sum(axis=0))))
        for name, Y in targets.items()
    }
    report["unfair_rate"] = {
        name: round(float((Y.sum(axis=1) > 0).mean()), 4) for name, Y in targets.items()
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
