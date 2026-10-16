"""Select equal numbers of harmful and benign examples before mixing them."""
import json
from pathlib import Path

import numpy as np
from datasets import concatenate_datasets, load_dataset

from .config import DATASET_REPO, DATASET_REVISION, RESOURCES


def load_pool(config, split="train", data_dir=None, repo=DATASET_REPO, revision=DATASET_REVISION):
    if data_dir is not None:
        files = sorted((Path(data_dir)/"data"/config).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No Parquet files in {Path(data_dir)/'data'/config}")
        return load_dataset("parquet", data_files={split:[str(p) for p in files]}, split=split)
    return load_dataset(repo, config, split=split, revision=revision)


def balanced_data(count=256, seed=0, safety_pool="legacy", data_dir=None,
                  repo=DATASET_REPO, revision=DATASET_REVISION):
    safety = load_pool("spavl_cleaned_train", data_dir=data_dir, repo=repo, revision=revision)
    utility = load_pool("vlsafe_benign_train", data_dir=data_dir, repo=repo, revision=revision)
    if safety_pool == "legacy":
        ids = json.loads((RESOURCES/"spavl_legacy_1000_ids.json").read_text(encoding="utf-8"))
        lookup = {key:i for i,key in enumerate(safety["sample_id"])}
        missing = set(ids)-set(lookup)
        if missing:
            raise ValueError(f"Historical SPA-VL IDs missing from pool: {sorted(missing)[:3]}")
        safety = safety.select([lookup[key] for key in ids])
    elif safety_pool != "full":
        raise ValueError("safety_pool must be legacy or full")
    if not 1 <= count <= min(len(safety),len(utility)):
        raise ValueError("samples-per-pool exceeds a pool size or is not positive")
    # Match the old BaseDataset.sample: reset NumPy seed independently per pool.
    selected = []
    for dataset,benign in ((safety,False),(utility,True)):
        indices = np.random.RandomState(seed).choice(len(dataset),count,replace=False)
        subset = dataset.select(indices)
        if benign:
            subset = subset.add_column("rejected",[""]*count)
        subset = subset.select_columns(["sample_id","question","chosen","rejected","image"])
        selected.append(subset.add_column("is_benign",[benign]*count))
    return concatenate_datasets(selected).shuffle(seed=seed)
