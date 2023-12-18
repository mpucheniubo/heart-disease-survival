from __future__ import annotations
from typing import Any, Callable

from copy import deepcopy
from dataclasses import dataclass
import logging
import pandas as pd
from pathlib import Path
import pickle
import re


@dataclass
class Split:
    """
    Small dataclass containing the information to split the data set into train and validation.
    """

    column: str = "dataset"
    size: float = 0.2
    train_label: str = "train"
    validation_label: str = "validation"

    def __post_init__(self) -> None:
        if self.size < 0.0 or self.size > 1.0:
            raise ValueError("Size must be between 0 and 1.")


class Base:
    """
    This class serves as a base to handel data structures. It has the following attributes/properties:

    data: `DataFrame`
            The data set for the instance.
    primary_key: `list[str]`
            List with columns of the data that represent a unique record.
    target: `str`
            The target column.
    event: `str`
            In survival analysis the censoring column.
    boolean_mapping: `dict[str, dict[str, bool]]`
            Dictionary with the mappings of the boolean columns in the data set.
    skip_columns: `list[str]`
            Read only property with columns to skip.
    split: `Split`
            Information necessary to split data set in train and validation.
    train_data: `DataFrame`
            Read only property with the train data set.
    validation_data: `DataFrame`
            Read only property with the validation data set.

    # Model

    ## __init__

    data: `DataFrame` or `None`, default `None`.
            Input containing the dataset. If `None` is provided, the respective attribute will be initialized with an empty `DataFrame`.
    primary_key: `list[str]` or `None`, default `None`.
            Input with the information about the primary key of the data set. If the inpute evaluates to `False`, the respective input will be initialized with an empty list.
    target: `str`, default `""`.
            Target column for prediction.
    event: `str`, default `""`.
            Event column for censoring in survival analysis.

    ## __bool__

    An object instance has value if the function `verify_primary_key` evaluates to `True` and the `data` attribute has rows.
    """

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        primary_key: list[str] | None = None,
        target: str = "",
        event: str = "",
    ) -> None:
        self.data: pd.DataFrame = data if data is not None else pd.DataFrame()
        self.primary_key: list[str] = primary_key or []
        self.target: str = target
        self.event: str = event
        self.boolean_mapping: dict[str, dict[str, bool]] = {}
        self.split: Split = Split()

    @property
    def skip_columns(self) -> list[str]:
        if self.event:
            return self.primary_key + [self.target, self.event, self.split.column]
        else:
            return self.primary_key + [self.target]

    def __bool__(self) -> None:
        _is = False
        if self.verify_primary_key() and self.data.shape[0] > 0:
            _is = True
        return _is

    @property
    def train_data(self) -> pd.DataFrame:
        if self.split.column not in self.data.columns:
            self.set_train_test_split()
        return self.data.loc[
            self.data[self.split.column] == self.split.train_label
        ].copy()

    @property
    def validation_data(self) -> pd.DataFrame:
        if self.split.column not in self.data.columns:
            self.set_train_test_split()
        return self.data.loc[
            self.data[self.split.column] == self.split.validation_label
        ].copy()

    class Decorators:
        @classmethod
        def verified(cls, func: Callable[[Any], Any]) -> Callable[[Any], Any]:
            def decorated(cls, *args, **kwargs):
                logging.info(
                    f"Applying decorator verified to method '{func.__name__}'."
                )
                value = None
                if cls:
                    value = func(cls, *args, **kwargs)
                return value

            return decorated

    def copy(self) -> Base:
        """
        This method provides a deepcopy of the object instance.

        # Return

        It outputs a new instance of the `Base` that has been copied.
        """
        logging.info("Creating copy of 'Base' instance.")
        return deepcopy(self)

    @staticmethod
    def _to_snake_case(name: str) -> str:
        """
        Protected method that converts a string to snake case. Solution found in https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case

        # Parameters

        name: `str`
            The string to make snake case.

        # Return

        It outputs a string in snake case.
        """
        logging.info(f"Setting column {name} to snake case.")
        name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        name = re.sub("__([A-Z])", r"_\1", name)
        name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
        return name.lower()

    @Decorators.verified
    def set_train_test_split(self, key: list[str] | None = None) -> None:
        """
        This adds a column to the data set with the split name and the respective labels.

        # Parameters

        key: `list[str]` or `None`, default `None`
            Set of columns to use as the key for the train/validation split. If no value is provided, the primary key will be used.
        """
        logging.info("Setting train test split.")
        if not key:
            key = self.primary_key.copy()
        df = self.data.sample(frac=1 - self.split.size, random_state=42)[key]
        df[self.split.column] = self.split.train_label
        self.data = self.data.merge(df, how="left", on=key)
        self.data[self.split.column].fillna(self.split.validation_label, inplace=True)

    def columns_to_snake_case(self) -> None:
        """
        This sets all columns to snake case, updating the target, event, primary key and split column attributes as well.
        """
        logging.info("Setting all data columns to snake case.")
        self.data.columns = self.data.columns.map(self._to_snake_case)
        self.target = self._to_snake_case(self.target)
        self.event = self._to_snake_case(self.event)
        self.primary_key = list(map(self._to_snake_case, self.primary_key))
        self.split.column = self._to_snake_case(self.split.column)

    def verify_primary_key(self) -> bool:
        """
        This method verifies that the primary key attribute represents in fact a unique record.

        # Return

        It outputs `True` if the primary key is verified, `False` otherwise.
        """
        logging.info("Verifying primary key of data.")
        return (
            self.data.drop_duplicates(subset=self.primary_key).shape[0]
            == self.data.shape[0]
        )

    @Decorators.verified
    def drop_columns_with_one_value(self) -> None:
        """
        This method removes columns that have only one value, excluding the columns of the primary key.
        """
        logging.info("Removing columns that have only one entry.")
        for column in self.data.columns:
            if self.data[column].nunique() == 1 and column not in self.primary_key:
                self.data.drop(columns=[column], inplace=True)
                logging.info(f"Column '{column}' has been removed.")

    @Decorators.verified
    def set_booleans(self) -> None:
        """
        This method sets binary columns to boolean dtype based on the object columns and stores the mapping in the boolean mapping attribute. Primary key columns are
        excluded.
        """
        logging.info("Setting binary columns to 'boolean'.")
        for column in self.data.select_dtypes(include="object"):
            if self.data[column].nunique() == 2 and column not in self.skip_columns:
                values = self.data[column].unique()
                self.boolean_mapping[column] = {True: values[0], False: values[1]}
                self.data[column] = (
                    self.data[column]
                    .map({values[0]: True, values[1]: False})
                    .astype("bool")
                )
                logging.info(f"Column '{column}' set to boolean.")

    @Decorators.verified
    def set_categories(self) -> None:
        """
        This method sets columns whose dtype is `object` to category if they have more than two different entries, excluding the columns of the primary key.
        """
        logging.info(
            "Setting columns with more than 2 different entries to 'category'."
        )
        for column in self.data.select_dtypes(include="object"):
            if self.data[column].nunique() > 2 and column not in self.skip_columns:
                self.data[column] = self.data[column].astype("category")
                logging.info(f"Column '{column}' set to category.")

    def make(self) -> None:
        """
        This method runs the main functions in the class to post-process the data.
        """
        self.columns_to_snake_case()
        self.set_train_test_split()
        self.drop_columns_with_one_value()
        self.set_booleans()
        self.set_categories()

    @Decorators.verified
    def save(self, path: Path, name: str) -> None:
        """
        This saves the object instance. The data set is stored separately as a parquet file.

        # Parameters

        path: `Path`
            Location of the stored object.
        name: `str`
            Complementary name of the instance.
        """
        logging.info("Attempting to save Base instance.")
        with open(path.joinpath(f"base_{name}.pkl"), "wb") as file:
            base = self.copy()
            base.data = pd.DataFrame(columns=base.data.columns)
            pickle.dump(base, file)
            self.data.to_parquet(path.joinpath(f"heart_{name}.parquet"), index=False)

    @staticmethod
    def load(path: Path, name: str) -> Base:
        """
        This loads the base instance with the respective dataset.

        # Parameters

        path: `Path`
            Location of the stored object.
        name: `str`
            Complementary name of the instance.

        # Return

        It outputs the loaded base instance with the data.
        """
        logging.info("Attempting to load Base instance.")
        with open(path.joinpath(f"base_{name}.pkl"), "rb") as file:
            base: Base = pickle.load(file)
            base.data = pd.read_parquet(path.joinpath(f"heart_{name}.parquet"))
            return base
