from __future__ import annotations

import fire
import logging

from models import Survival

logging.basicConfig(level=logging.INFO)


def main(
    name: str = "xgbse",
    model: str = "XGBSEKaplanNeighborsModel",
    use_cached: bool = False,
) -> None:
    """
    This is the main function to train the model, compute the metrics and make the predictions with the respective images.

    If the process has been run once, one can re-run it with `use_cached` set to `True` so everything gets loaded instead of recomputing features and retraining the model. Note
    that there is no fail-safe for this option, so if it is run without the expected files being there, the load will crash the program.

    # Parameters

    name: `str`, default `"xgbse"`
        Name to use to save the different cached instances.
    model: `str`, default `"XGBSEKaplanNeighborsModel"`
        Name of the model to use.
    use_cached: `bool`, default `False`
        Whether to use cached instances or not.
    """
    if use_cached:
        survival = Survival.load(name)
    else:
        survival = Survival.create(model).make_features().train_and_evaluate()
        survival.save(name)

    logging.info(survival.model.metrics)

    logging.info(survival.model.feature_importance)

    survival.make_predictions()
    survival.make_figs()


if __name__ == "__main__":
    fire.Fire(main)
