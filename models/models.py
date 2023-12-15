from __future__ import annotations

from abc import ABC, abstractmethod

from xgbse import XGBSEKaplanTree

from models.features import Features


class Model(ABC):
    """
    Abstract base class to create an interface for all models.
    """

    @abstractmethod
    def __init__(self, features: Features) -> None:
        self._features: Features = features

    @abstractmethod
    def train(self) -> None:
        ...

    @abstractmethod
    def predict(self) -> None:
        ...

    @abstractmethod
    def save(self) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(self) -> Model:
        ...


class XGBSEKaplanTreeModel(Model):
    def __init__(self, features: Features) -> None:
        super().__init__(features)
        self.model: XGBSEKaplanTree = XGBSEKaplanTree()

    def train(self) -> None:
        self.model.fit()

    def predict(self) -> None:
        return super().predict()

    def save(self) -> None:
        return super().save()

    @classmethod
    def load() -> XGBSEKaplanTreeModel:
        ...
