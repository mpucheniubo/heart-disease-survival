from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.base import Base

@pytest.fixture(scope="session")
def base() -> Base:
    np.random.seed(42)
    size = 100
    a = np.random.randint(0, 100, size=size)
    b = 100 * np.random.rand(size)
    c = np.ones(size)
    d = np.random.choice(["X", "Y", "Z"], size=size)
    e = np.random.choice(["p", "q", "r", "s", "t"], size=size)
    f = np.random.choice(["Y", "N"], size=size)
    g = np.random.choice(["0", "1"], size=size)
    data = pd.DataFrame(data={"a":a, "b":b, "c":c, "d":d, "e":e, "f":f, "g":g}, index=np.arange(size))
    data.reset_index(inplace=True)
    data.rename(columns={"index":"id"}, inplace=True)
    return Base(data=data, primary_key=["id"])