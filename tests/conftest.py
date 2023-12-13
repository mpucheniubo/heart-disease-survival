from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.base import Base


@pytest.fixture(scope="session")
def base() -> Base:
    """
    This provides a `Base` instance with a dataset.
    """
    np.random.seed(42)
    size = 100
    a = np.random.randint(0, 100, size=size)
    b = 100 * np.random.rand(size)
    c = np.ones(size)
    d = np.random.choice(["X", "Y", "Z"], size=size)
    e = np.random.choice(["p", "q", "r", "s", "t"], size=size)
    f = np.random.choice(["Y", "N"], size=size)
    g = np.random.choice(["0", "1"], size=size)
    data = pd.DataFrame(
        data={"a": a, "b": b, "c": c, "d": d, "e": e, "f": f, "g": g},
        index=np.arange(size),
    )
    data.reset_index(inplace=True)
    data.rename(columns={"index": "id"}, inplace=True)
    return Base(data=data, primary_key=["id"])


@pytest.fixture(scope="session")
def data() -> pd.DataFrame:
    """
    This provides a data set to test the features.
    """
    SIZE = 20
    np.random.seed(42)
    data = pd.DataFrame(
        data={
            "id": np.arange(SIZE),
            "age": np.random.randint(1, 80, size=SIZE),
            "cholesterol": np.random.randint(70, 300, size=SIZE),
        }
    )
    return data
