from __future__ import annotations
from typing import Any, Callable

from copy import deepcopy
import logging
import pandas as pd
import re

logging.basicConfig(level=logging.INFO)


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

    def __bool__(self) -> None:
        _is = False
        if self.verify_primary_key() and self.data.shape[0] > 0:
            _is = True
        return _is

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
        Protected method that converst a string to snake case. Solution found in https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case

        # Parameters

        name: `str`
            The string to make snake case.

        # Return

        It outputs a string in snake case.
        """
        name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        name = re.sub("__([A-Z])", r"_\1", name)
        name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
        return name.lower()

    def columns_to_snake_case(self) -> None:
        self.data.columns = self.data.columns.map(self._to_snake_case)
        self.target = self._to_snake_case(self.target)
        self.event = self._to_snake_case(self.event)
        self.primary_key = list(map(self._to_snake_case, self.primary_key))

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
            if self.data[column].nunique() == 2 and column not in self.primary_key:
                values = self.data[column].unique()
                self.boolean_mapping[column] = {values[0]: True, values[1]: False}
                self.data[column] = (
                    self.data[column].map(self.boolean_mapping[column]).astype("bool")
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
            if self.data[column].nunique() > 2 and column not in self.primary_key:
                self.data[column] = self.data[column].astype("category")
                logging.info(f"Column '{column}' set to category.")

    def make(self) -> None:
        """
        This method runs the main functions in the class to post-process the data.
        """
        self.columns_to_snake_case()
        self.drop_columns_with_one_value()
        self.set_booleans()
        self.set_categories()
