from __future__ import annotations

from abc import ABC, abstractmethod


class Model(ABC):
    """
    Abstract base class to create an interface for all models.
    """

    @abstractmethod
    def __init__(self) -> None:
        ...

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
