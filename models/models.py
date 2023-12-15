from __future__ import annotations
from typing import Any

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from sksurv.metrics import integrated_brier_score
from xgbse import XGBSEKaplanTree
from xgbse.converters import convert_to_structured
from xgbse.metrics import approx_brier_score

from models.features import Features


class Model(ABC):
    """
    Abstract base class to create an interface for all models.
    """

    @abstractmethod
    def __init__(self, features: Features) -> None:
        self._features: Features = features
        self._model: Any = None
        self.metrics: pd.DataFrame = pd.DataFrame(
            columns=[
                "model",
                "metric",
                self._features._base.split.train_label,
                self._features._base.split.validation_label,
            ]
        )

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def train(self) -> None:
        ...

    @abstractmethod
    def predict(self) -> None:
        ...

    @abstractmethod
    def save(self, path: Path, name: str) -> None:
        with open(path.joinpath(f"{name}.pkl"), "wb") as file:
            pickle.dump(self._model, file)
            self.metrics.to_parquet(
                path.joinpath(f"metrics_{name}.parquet"), index=False
            )

    @abstractmethod
    def compute_metrics() -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path, name: str, features: Features) -> Model:
        with open(path.joinpath(f"{name}.pkl"), "rb") as file:
            model = cls(features)
            model._model = pickle.load(file)
            model.metrics = pd.read_parquet(path.joinpath(f"metrics_{name}.parquet"))
            return model


class XGBSEKaplanTreeModel(Model):
    def __init__(
        self, features: Features, params: dict[str, Any] | None = None
    ) -> None:
        super().__init__(features)
        self._model: XGBSEKaplanTree = XGBSEKaplanTree(params)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def time_bins(self) -> np.ndarray:
        return np.arange(
            self._features._base.train_data[self._features._base.target].max()
        )

    @property
    def x_train(self) -> pd.DataFrame:
        return self._features._base.train_data[
            self._features.get_categorical() + self._features.get_numerical()
        ]

    @property
    def y_train(self) -> np.ndarray:
        y = self._features._base.train_data[
            [self._features._base.target, self._features._base.event]
        ]
        return convert_to_structured(
            y[self._features._base.target].astype("float"),
            y[self._features._base.event].astype("bool"),
        )

    @property
    def x_validation(self) -> pd.DataFrame:
        return self._features._base.validation_data[
            self._features.get_categorical() + self._features.get_numerical()
        ]

    @property
    def y_validation(self) -> np.ndarray:
        y = self._features._base.validation_data[
            [self._features._base.target, self._features._base.event]
        ]
        return convert_to_structured(
            y[self._features._base.target].astype("float"),
            y[self._features._base.event].astype("bool"),
        )

    def train(self) -> None:
        self._model.fit(self.x_train, self.y_train, time_bins=self.time_bins)

    def predict(self, x: pd.DataFrame | None = None) -> None:
        if x is None:
            x = self._features._base.data[
                self._features.get_categorical() + self._features.get_numerical()
            ]
        return self._model.predict(x)

    def save(self, path: Path, name: str) -> None:
        super().save(path, name)

    def compute_metrics(self) -> None:
        train_preds = self.predict(self.x_train)
        validation_preds = self.predict(self.x_validation)
        self._compute_xgbse_ibs(train_preds, validation_preds)
        self._compute_sksurv_ibs(train_preds, validation_preds)

    def _compute_sksurv_ibs(
        self, train_preds: pd.DataFrame, validation_preds: pd.DataFrame
    ) -> None:
        time_bins = np.arange(self.y_train["c2"].min(), self.y_train["c2"].max())
        ibs_train = integrated_brier_score(
            self.y_train, self.y_train, train_preds[time_bins], time_bins
        )
        time_bins = np.arange(
            self.y_validation["c2"].min(), self.y_validation["c2"].max()
        )
        ibs_validation = integrated_brier_score(
            self.y_validation, self.y_validation, validation_preds[time_bins], time_bins
        )
        self.metrics = pd.concat(
            [
                self.metrics,
                pd.DataFrame(
                    data={
                        "model": self.name,
                        "metric": "sksurv_ibs",
                        self._features._base.split.train_label: ibs_train,
                        self._features._base.split.validation_label: ibs_validation,
                    },
                    index=[0],
                ),
            ]
        ).reset_index(drop=True)

    def _compute_xgbse_ibs(
        self, train_preds: pd.DataFrame, validation_preds: pd.DataFrame
    ) -> None:
        ibs_train = approx_brier_score(self.y_train, train_preds)
        ibs_validation = approx_brier_score(self.y_validation, validation_preds)
        self.metrics = pd.concat(
            [
                self.metrics,
                pd.DataFrame(
                    data={
                        "model": self.name,
                        "metric": "xgbse_ibs",
                        self._features._base.split.train_label: ibs_train,
                        self._features._base.split.validation_label: ibs_validation,
                    },
                    index=[0],
                ),
            ]
        ).reset_index(drop=True)

    @classmethod
    def load(cls, path: Path, name: str, features: Features) -> XGBSEKaplanTreeModel:
        return super().load(path, name, features)
