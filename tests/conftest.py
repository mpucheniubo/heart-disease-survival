from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.base import Base

@pytest.fixture(scope="session")
def base() -> Base:
    size = 100
    a = np.random.randint(0, 100, size=size)
    b = 100 * np.random.rand(size)
    data = pd.DataFrame(data={"a":a, "b":b}, index=np.arange(size))
    data.reset_index(inplace=True)
    data.rename(columns={"index":"id"}, inplace=True)
    return Base(data=data, primary_key=["id"])