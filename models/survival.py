from __future__ import annotations

import logging
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os

from models.base import Base
from models.features import Features
from models.models import Model, XGBSEKaplanTreeModel
from models.utils import classproperty


class Survival:
    """
    This is the main class to produce the predictions with survival analysis for the data set. The attributes/properties are:

    PATH: `Path`
        Root directory.
    MODEL_MAPPING: `dict[str, Model]`
        Mapping with the different models, name to class.
    base: `Base`
        Main structure to manage and manipulate the data.
    features: `Features`
        Main structure to create and add futures to the data set.
    model: `Model`
        The chosen model to use for the predictions.
    predictions: `DataFrame`
        Resulting predictions.
    base_path: `Path`
        Path to store base instances.
    features_path: `Path`
        Path to store features instances.
    figs_path: `Path`
        Path to store figures.
    model_path: `Path`
        Path to store model instances.
    """

    PATH: Path = Path(__file__).parent.parent

    MODEL_MAPPING: dict[str, Model] = {
        "XGBoostSurvivalEmbeddings": XGBSEKaplanTreeModel
    }

    def __init__(self, base: Base, model: str = "XGBoostSurvivalEmbeddings") -> None:
        self.base: Base = base
        self.features: Features = Features(base=self.base)
        self.model: Model = self.MODEL_MAPPING.get(model)(features=self.features)
        self.predictions: pd.DataFrame = pd.DataFrame()

    @classproperty
    def base_path(self) -> Path:
        return self.PATH.joinpath("data", "base")

    @classproperty
    def features_path(self) -> Path:
        return self.PATH.joinpath("data", "features")

    @property
    def figs_path(self) -> Path:
        return self.PATH.joinpath("data", "figs")

    @classproperty
    def model_path(self) -> Path:
        return self.PATH.joinpath("data", "model")

    @classmethod
    def create(cls, model: str = "XGBoostSurvivalEmbeddings") -> Survival:
        """
        Method to instantiate an object tailored to the problem at hand.

        # Parameters

        model: `str`
            Name of the model to use.

        # Returen

        It outputs the instantiated object.
        """
        logging.info("Creating Survival instance from CSV file.")
        data = (
            pd.read_csv(
                cls.PATH.joinpath("data", "csv", "heart.csv"), sep=",", decimal="."
            )
            .reset_index()
            .rename(columns={"index": "id"})
        )
        survival = cls(
            Base(data=data, primary_key=["id"], target="age", event="target"),
            model,
        )
        survival.base.make()

        return survival

    @classmethod
    def load(cls, name: str, model: str = "XGBoostSurvivalEmbeddings") -> Survival:
        """
        Method to load a precomputed model and data with features.

        # Parameters

        name: `str`
            Complementary name of the instance.
        model: `str`
            Name of the model to use.

        # Return

        It outputs the instantiated, loaded object.
        """
        logging.info("Attempting to load Survival instance.")
        base = Base.load(cls.base_path, name)
        survival = cls(base)
        survival.features = Features.load(cls.features_path, name, survival.base)
        survival.model = cls.MODEL_MAPPING.get(model).load(
            cls.model_path, name, survival.features
        )
        return survival

    def make_features(self, use_ohe: bool = False) -> Survival:
        """
        Makes the features for the data set, with the option to use one hot encoding for categorical variables.

        # Parameters

        use_ohe: `bool`, default `False`
            Whether to use OHE or not.

        # Return

        It outputs the same instance for concatenation.
        """
        logging.info("Making all features.")
        # cols to numerical
        num_cols = [
            col
            for col in self.base.data.select_dtypes(include=["float", "int"])
            if col not in self.base.skip_columns
        ]
        self.features.to_numerical(columns=num_cols, keep_original=True)

        # cols to categorical
        cat_cols = [
            col
            for col in self.base.data.select_dtypes(include=["bool", "category"])
            if col not in self.base.skip_columns
        ]
        self.features.to_categorical(
            columns=cat_cols, keep_original=True, use_ohe=use_ohe
        )

        # compute features
        self.features.cholesterol_threshold()
        self.features.max_theoretical_heart_rate()

        return self

    def train_and_evaluate(self) -> Survival:
        """
        This trains and computes the error metrics for the model.

        # Return

        It outputs the same instance for concatenation.
        """
        logging.info("Training model and computing metrics.")
        self.model.train()
        self.model.compute_metrics()

        return self

    def save(self, name: str) -> Survival:
        """
        This stores the necessary objects to reconstruct the object instance.

        # Parameters

        name: `str`
            Complementary name of the instance.

        # Return

        It outputs the same instance for concatenation.
        """
        logging.info("Attempting to save Survival instance.")
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.features_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)

        self.base.save(self.base_path, name)
        self.features.save(self.features_path, name)
        self.model.save(self.model_path, name)

        return self

    def make_predictions(self) -> Survival:
        """
        This populates the predictions data frame with the predictions and makes some manipulations to provide the
        primary key together with the ages and respective survival rates.

        # Return

        It outputs the same instance for concatenation.
        """
        logging.info("Running predictions for the whole data set.")
        self.predictions = self.model.predict()
        columns = self.predictions.columns
        self.predictions[self.base.primary_key] = self.base.data[self.base.primary_key]
        self.predictions = self.predictions.melt(
            id_vars=self.base.primary_key,
            value_vars=columns,
            var_name="age",
            value_name="survival_rate",
        )
        self.predictions.sort_values(by=self.base.primary_key + ["age"], inplace=True)
        self.predictions.reset_index(drop=True, inplace=True)
        return self

    def make_figs(self) -> Survival:
        """
        This orchestrates the production of the figures.

        # Return

        It outputs the same instance for concatenation.
        """
        self._fig_overall_survival()
        self._fig_sex_survival()
        return self

    def _fig_overall_survival(self) -> None:
        """
        This creates a figure for the average survival rate of the whole data set.
        """
        df = self.predictions.groupby("age")[["survival_rate"]].mean().reset_index()
        plt.plot(df["age"], df["survival_rate"], "b-")
        plt.xlabel("Age [Years]")
        plt.ylabel("Survival rate")
        plt.title("Heart Disease Survival Probability")
        plt.tight_layout()
        plt.savefig(self.figs_path.joinpath("survival_rates.png"))
        plt.show()

    def _fig_sex_survival(self) -> None:
        """
        This creates a figure for the average survival rate of the whole data set per sex.
        """
        df = self.predictions.merge(
            self.base.data[self.base.primary_key + ["sex"]],
            on=self.base.primary_key,
            how="left",
        )
        df["sex"] = df["sex"].map(self.base.boolean_mapping.get("sex"))

        df = df.groupby(["age", "sex"])[["survival_rate"]].mean().reset_index()
        for sex, _df in df.groupby("sex"):
            plt.plot(_df["age"], _df["survival_rate"], label=f"Sex [{sex}]")

        plt.legend(loc=1)
        plt.xlabel("Age [Years]")
        plt.ylabel("Survival rate")
        plt.title("Heart Disease Survival Probability (Sex)")
        plt.tight_layout()
        plt.savefig(self.figs_path.joinpath("survival_rates_sex.png"))
        plt.show()
