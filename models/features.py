from __future__ import annotations
from typing import ClassVar

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pandas.core.groupby.groupby import GroupBy
from sklearn.preprocessing import OneHotEncoder

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
    This is an abstraction layer to compute all the features using the `Base`. It has the following properties/attributes:

    _base: `Base`
        Protected attribute with the `Base` instance to work with.
    one_hot_encoder: `OneHotEncoder`
        Initialized one hot encoder.
    marker: `str`
        Read only property with the prefix for a feature.
    categorical_marker: `str`
        Read only property with the prefix for a categorical feature.
    numerical_marker: `str`
        Read only property with the prefix for a numerical feature.
    """

    def __init__(self, base: Base) -> None:
        self._base: Base = base
        self.one_hot_encoder: OneHotEncoder = OneHotEncoder(handle_unknown="ignore")

    @property
    def marker(self) -> str:
        return Feature.marker

    @property
    def categorical_marker(self) -> str:
        return Feature.categorical_marker

    @property
    def numerical_marker(self) -> str:
        return Feature.numerical_marker

    def get_numerical(self) -> list[str]:
        """
        It provides a list with the columns corresponding to the numerical features.

        # Return

        List whose entries start with the numerical marker.
        """
        feats = self._base.data.columns[
            self._base.data.columns.str.startswith(self.numerical_marker)
        ]
        return list(feats)

    def get_categorical(self) -> list[str]:
        """
        It provides a list with the columns corresponding to the categorical features.

        # Return

        List whose entries start with the categorical marker.
        """
        feats = self._base.data.columns[
            self._base.data.columns.str.startswith(self.categorical_marker)
        ]
        return list(feats)

    def to_numerical(self, columns: list[str], keep_original: bool = False) -> None:
        """
        This adds the numerical feature prefix to the provided columns.

        # Parameters

        columns: `list[str]`
            List of the columns to mark as numerical features.
        keep_original: `bool`, default `False`
            Whether to keep the original columns or not.
        """
        column_mapping = {column: self.numerical_marker + column for column in columns}
        if keep_original:
            for column, fet_column in column_mapping.items():
                self._base.data[fet_column] = self._base.data[column]
        else:
            self._base.data.rename(columns=column_mapping, inplace=True)

    def to_categorical(
        self, columns: list[str], keep_original: bool = False, use_ohe: bool = False
    ) -> None:
        """
        This adds the categorical feature prefix to the provided columns and provides numerical values instead of the category entities. If `use_ohe` is set
        to `False`, an enumeration of the categories is given as the feature. Otherwise, a one hot encoding is performed and the encoder stored in the property
        `one_hot_encoder`.

        # Parameters

        columns: `list[str]`
            List of the columns to mark as numerical features.
        keep_original: `bool`, default `False`
            Whether to keep the original columns or not.
        use_ohe: `bool`, default `False`
            Whether to use a one hot encoding or not.
        """
        column_mapping = {
            column: self.categorical_marker + column for column in columns
        }
        if use_ohe:
            self.one_hot_encoder.fit(self._base.data[column_mapping.keys()])
            feat_columns = [
                self.categorical_marker + column
                for column in self.one_hot_encoder.get_feature_names_out()
            ]
            feats = pd.DataFrame(
                self.one_hot_encoder.transform(
                    self._base.data[column_mapping.keys()].values
                ).toarray(),
                columns=feat_columns,
            )
            self._base.data = pd.concat([self._base.data, feats], axis=1)
            self._base.columns_to_snake_case()
        else:
            for column, feat_column in column_mapping.items():
                df = self._base.data[[column]].drop_duplicates()
                df[feat_column] = np.arange(df.shape[0])
                self._base.data = self._base.data.merge(df, how="left", on=[column])

        if not keep_original:
            self._base.data.drop(columns=column_mapping.keys(), inplace=True)

    def cholesterol_threshold(self) -> None:
        """
        This creates the feature to determine whether the cholesterol goes above the threshold or not.
        """
        feature = FeatureCholesterolThreshold()
        self._base.data[feature.name] = feature.run(
            column=f"{self.numerical_marker}cholesterol", data=self._base.data
        ).astype("int")

    def max_theoretical_heart_rate(self) -> None:
        """
        This creates the feature with the max heart rate estimate.
        """
        feature = FeatureMaxTheoreticalHR()
        self._base.data[feature.name] = feature.run(column="age", data=self._base.data)
