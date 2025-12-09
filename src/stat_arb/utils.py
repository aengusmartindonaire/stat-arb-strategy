from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set global random seeds for numpy and Python's random module.
    Call this at the start of experiments for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    """
    Make sure a directory exists (like mkdir -p) and return it as a Path.

    Useful when saving outputs, e.g.:

        out_dir = ensure_dir("outputs/plots")
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def chunk_iterable(xs: Iterable, chunk_size: int):
    """
    Yield chunks of size `chunk_size` from an iterable.

    Handy when you want to process tickers in batches.
    """
    chunk = []
    for x in xs:
        chunk.append(x)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk