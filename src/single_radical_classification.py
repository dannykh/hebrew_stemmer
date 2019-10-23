from copy import deepcopy
from functools import reduce
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src.evaluation import run_test
from src.features import baseline_pipeline_steps
from src.utils import ith_radical_getter


class SimpleRadicalCombination:
    def __init__(self, pipeline_steps: List, verbose=True):
        self.radical_models = []
        self.radical_models_scores = []
        self.pipeline_steps = pipeline_steps
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y):
        self.radical_models = []
        self.radical_models_scores = []
        for radical_i in range(0, 3):
            # Evaluate cross val score of i'th radical model
            model_i = Pipeline(deepcopy(self.pipeline_steps))
            self.radical_models_scores.append(np.average(
                cross_val_score(model_i, X, y.apply(ith_radical_getter(radical_i)), cv=10)))

            if self.verbose:
                print(f"Cross val score for radical {radical_i + 1} : {self.radical_models_scores[-1]}")

            # Train model to predict i'th radical on train set
            model_i.fit(X, y.apply(ith_radical_getter(radical_i)))
            self.radical_models.append(model_i)

        return self

    def predict(self, X: pd.DataFrame):
        radical_predictions = [model.predict(X) for model in self.radical_models]
        return reduce(lambda x, y: x + y, radical_predictions)


run_test(SimpleRadicalCombination(baseline_pipeline_steps), "Simple radical combination")
