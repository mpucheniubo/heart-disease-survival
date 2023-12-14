from __future__ import annotations

from abc import ABC, abstractmethod

from xgbse import XGBSEKaplanTree

from models.base import Base


class Model(ABC):
    """
    Abstract base class to create an interface for all models.
    """

    @abstractmethod
    def __init__(self, base: Base) -> None:
        self._base: Base = base

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
    def __init__(self, base: Base) -> None:
        super().__init__(base)
        self.model: XGBSEKaplanTree = XGBSEKaplanTree()

    def train(self) -> None:
        return super().train()

    def predict(self) -> None:
        return super().predict()

    def save(self) -> None:
        return super().save()

    @classmethod
    def load() -> XGBSEKaplanTreeModel:
        ...
