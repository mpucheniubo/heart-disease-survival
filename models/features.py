from __future__ import annotations
from typing import ClassVar

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pandas.core.groupby.groupby import GroupBy

from .base import Base
from .utils import classproperty


@dataclass
class Feature(ABC):
    """
    This is an `ABC` for the individual features.
    """

    marker: ClassVar[str] = "f|"

    @classproperty
    def categorical_marker(self) -> str:
        return Feature.marker + "cat|"

    @classproperty
    def numerical_marker(self) -> str:
        return Feature.marker + "num|"

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Read-only property with the name of the resulting feature.
        """
        ...

    @abstractmethod
    def run(
        self, column: str, data: pd.DataFrame | GroupBy
    ) -> pd.Series | pd.DataFrame:
        """
        Main method to compute the feature.
        """
        ...


@dataclass
class FeatureCholesterolThreshold(Feature):
    """
    This feature determines whether the cholesterol is above a given threshold or not.
    """

    threshold: float = 150

    @property
    def name(self) -> str:
        return self.categorical_marker + "cholesterol_threshold"

    def run(self, column: str, data: pd.DataFrame) -> pd.Series:
        return np.where(data[column] >= self.threshold, True, False).astype("bool")


@dataclass
class FeatureMaxTheoreticalHR(Feature):
    """
    This feature computes the theoretical max HR by using the infamous formul `220 - AGE`.
    """

    threshold: int = 220

    @property
    def name(self) -> str:
        return self.numerical_marker + "max_theoretical_hr"

    def run(self, column: str, data: pd.DataFrame) -> pd.Series:
        return self.threshold - data[column]


class Features:
    """
    This is an abstraction layer to compute all the features using the `Base`.
    """

    def __init__(self, base: Base) -> None:
        self._base: Base = base

    @property
    def marker(self) -> str:
        return Feature.marker

    @property
    def categorical_marker(self) -> str:
        return Feature.categorical_marker

    @property
    def numerical_marker(self) -> str:
        return Feature.numerical_marker
