from __future__ import annotations
from typing import Literal

import pandas as pd
from pathlib import Path

from models.base import Base
from models.features import Features
from models.models import Model, XGBSEKaplanTreeModel


class Survival:
    PATH: Path = Path(__file__).parent.parent

    MODEL_MAPPING: dict[str, Model] = {
        "XGBoostSurvivalEmbeddings": XGBSEKaplanTreeModel
    }

    def __init__(self, base: Base, model: str = "XGBoostSurvivalEmbeddings") -> None:
        self.base: Base = base
        self.features: Features = Features(base=self.base)
        self.model: Model = self.MODEL_MAPPING.get(model)()

    @classmethod
    def create(cls) -> Survival:
        data = (
            pd.read_csv(cls.PATH.joinpath("data", "heart.csv"), sep=",", decimal=".")
            .reset_index()
            .rename(columns={"index": "Id"})
        )
        survival = cls(
            Base(data=data, primary_key=["Id"], target="Age", event="HeartDisease")
        )
        survival.base.make()

        return survival

    @classmethod
    def load(cls) -> Survival:
        ...

    def make_features(self, use_ohe: bool = False) -> Survival:
        # cols to numerical
        num_cols = [
            col
            for col in self.base.data.select_dtypes(include=["float", "int"])
            if col not in self.base.skip_columns
        ]
        self.features.to_numerical(columns=num_cols)

        # cols to categorical
        cat_cols = [
            col
            for col in self.base.data.select_dtypes(include=["bool", "category"])
            if col not in self.base.skip_columns
        ]
        self.features.to_categorical(columns=cat_cols, use_ohe=use_ohe)

        # compute features
        self.features.cholesterol_threshold()
        self.features.max_theoretical_heart_rate()

        return self
