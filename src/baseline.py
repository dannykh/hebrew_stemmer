import pickle
from copy import deepcopy
from functools import reduce
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from src.features import morpheme_breakdown, pos_encoder, \
    pattern_encoder, prefix_encoder
from src.parser import split_corpus_and_roots
from src.utils import ith_radical_getter
from sklearn.ensemble import RandomForestClassifier

baseline_pipeline_steps = [

    ("features", FeatureUnion([
        ("morpheme breakdown", morpheme_breakdown),
        ("pos encoder", pos_encoder),
        ("pattern encoder", pattern_encoder),
        ("prefix encoder", prefix_encoder),

    ])),

    # Use an SVC classifier
    # ("SVC", LinearSVC())
    # ("LogLinear",LogisticRegression(multi_class="auto",solver="liblinear"))
    ("random forest", RandomForestClassifier(n_estimators=100)),
]

# Load corpus :
# combined_train: pd.DataFrame = load_corpus(CSV_PATH).fillna("").drop_duplicates()

with open("../data/corpus_divided_reg_irreg.pkl", "rb") as fp:
    regular, irregular = pickle.load(fp)

test_prop = 0.2
regular_train, regular_test = train_test_split(regular, test_size=test_prop)
irregular_train, irregular_test = train_test_split(irregular, test_size=test_prop)

combined_train = pd.concat((regular_train, irregular_train))
combined_test = pd.concat((regular_test, irregular_test))
corpus, roots = split_corpus_and_roots(combined_train)


# Whole root prediction :
# whole_root_score = np.average(cross_val_score(baseline_pipeline, corpus, roots, cv=10))
# print(f"Cross val score for whole root : {whole_root_score}")


class SimpleRadicalCombination:
    def __init__(self, pipeline_steps: List, verbose=True):
        self.radical_models = []
        self.radical_models_scores = []
        self.pipeline_steps = pipeline_steps
        self.verbose = verbose

    def fit(self, train_corpus: pd.DataFrame):
        y = train_corpus.root
        X = train_corpus.drop("root", 1)
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


def run_eval(trained_model, target_sets):
    for target_set_desc, target_set in target_sets:
        score = (trained_model.predict(target_set) == target_set.root).value_counts()[True] / len(
            target_set)
        print(f"Model score on {target_set_desc} test set : {score}")


base = SimpleRadicalCombination(baseline_pipeline_steps).fit(combined_train)

run_eval(base, [("regular", regular_test), ("irregular", irregular_test), ("combined", combined_test)])
