from __future__ import annotations

import pandas as pd
from pathlib import Path
import os

from models.base import Base
from models.features import Features
from models.models import Model, XGBSEKaplanTreeModel
from models.utils import classproperty


class Survival:
    PATH: Path = Path(__file__).parent.parent

    MODEL_MAPPING: dict[str, Model] = {
        "XGBoostSurvivalEmbeddings": XGBSEKaplanTreeModel
    }

    def __init__(self, base: Base, model: str = "XGBoostSurvivalEmbeddings") -> None:
        self.base: Base = base
        self.features: Features = Features(base=self.base)
        self.model: Model = self.MODEL_MAPPING.get(model)(features=self.features)

    @classproperty
    def base_path(self) -> Path:
        return self.PATH.joinpath("data", "base")

    @classproperty
    def model_path(self) -> Path:
        return self.PATH.joinpath("data", "model")

    @classmethod
    def create(cls, model: str = "XGBoostSurvivalEmbeddings") -> Survival:
        data = (
            pd.read_csv(
                cls.PATH.joinpath("data", "csv", "heart.csv"), sep=",", decimal="."
            )
            .reset_index()
            .rename(columns={"index": "Id"})
        )
        survival = cls(
            Base(data=data, primary_key=["Id"], target="Age", event="HeartDisease"),
            model,
        )
        survival.base.make()

        return survival

    @classmethod
    def load(cls, name: str, model: str = "XGBoostSurvivalEmbeddings") -> Survival:
        base = Base.load(cls.base_path, name)
        survival = cls(base)
        survival.model = cls.MODEL_MAPPING.get(model).load(
            cls.model_path, name, survival.features
        )
        return survival

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

    def train_and_evaluate(self) -> Survival:
        self.model.train()
        self.model.compute_metrics()

        return self

    def save(self, name: str) -> Survival:
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)

        self.base.save(self.base_path, name)
        self.model.save(self.model_path, name)

        return self
