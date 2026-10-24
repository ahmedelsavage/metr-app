import json
from typing import cast

import pandas as pd
from datasets import load_dataset

from ml.config import CONFIG, DATASET, LABEL_NAMES_FILE, RAW_DIR, SPLITS


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    label_names = None
    for name, hf_split in SPLITS.items():
        ds = load_dataset(DATASET, CONFIG, split=hf_split)
        cast(pd.DataFrame, ds.to_pandas()).to_csv(RAW_DIR / f"{name}.csv", index=False)
        label_names = ds.features["labels"].feature.names
        print(f"{name}: {len(ds)} rows")
    LABEL_NAMES_FILE.write_text(json.dumps(label_names, indent=2))
    print("labels:", label_names)


if __name__ == "__main__":
    main()
