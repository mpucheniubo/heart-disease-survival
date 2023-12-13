from __future__ import annotations

import numpy as np
import pandas as pd

from models.features import FeatureCholesterolThreshold, FeatureMaxTheoreticalHR


def test_cholesterol_threshold(data: pd.DataFrame) -> None:
    """
    This tests the feature that determines whether the cholesterol is above a given threshold.
    """
    feature = FeatureCholesterolThreshold()
    data[feature.name] = feature.run("cholesterol", data)

    # alternative calculation of cholesterol threshold
    values = np.zeros(data.shape[0]).astype("bool")
    for _, value in data.iterrows():
        if value["cholesterol"] >= feature.threshold:
            values[value["id"]] = True

    np.testing.assert_array_equal(values, data[feature.name].values)


def test_max_theoretical_hr(data: pd.DataFrame) -> None:
    """
    This tests the feature that computes a theoretical max rate.
    """
    feature = FeatureMaxTheoreticalHR()
    data[feature.name] = feature.run("age", data)

    # alternative calculation of the max theoretical HR
    values = np.zeros(data.shape[0])
    for _, value in data.iterrows():
        values[value["id"]] = feature.threshold - value["age"]

    np.testing.assert_array_equal(values, data[feature.name].values)
