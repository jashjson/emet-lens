import numpy as np
import pandas as pd

COLUMNS = ["path", "label", "type", "source_id", "split"]  # label: real|fake; type: e.g. real, stylegan, diffusion, faceswap, tts


def load_manifest(path="data/MANIFEST.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"MANIFEST missing columns {missing}")
    # Leakage guard: a source_id must live in exactly one split.
    bad = df.groupby("source_id")["split"].nunique()
    if (bad > 1).any():
        raise ValueError(f"source_id appears in multiple splits: {list(bad[bad > 1].index[:5])}")
    return df


def assign_splits(df: pd.DataFrame, test_frac=0.3, seed=0) -> pd.DataFrame:
    """Assign train/test by source_id (never by frame)."""
    rng = np.random.RandomState(seed)
    ids = np.array(sorted(df["source_id"].unique()))
    rng.shuffle(ids)
    test = set(ids[: int(len(ids) * test_frac)])
    df = df.copy()
    df["split"] = df["source_id"].map(lambda s: "test" if s in test else "train")
    return df
