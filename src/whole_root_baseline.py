from functools import reduce
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.svm import LinearSVC

from src.features import ColumnSelector, StringDataSplit, heb_alphabet, RowRemover
from src.parser import load_corpus, CSV_PATH, split_corpus_and_roots
from src.utils import ith_radical_getter

max_word_len = 11

# Feature extractors :

morpheme_breakdown = Pipeline([
    # Select raw features (remove POS)
    ("select raw data", ColumnSelector(["morpheme"])),

    # Expand morpheme to feature for every letter
    ("expand morpheme", StringDataSplit("morpheme", max_word_len, "morph")),

    # Encode categorical features
    ("encode", OneHotEncoder(sparse=False, categories=[heb_alphabet for _ in range(0, max_word_len)]))
])

pos_encoder = Pipeline([
    ("select POS", ColumnSelector(["pos"])),

    ("encode", OneHotEncoder(sparse=False))
])

pattern_encoder = Pipeline([
    ("select pattern", ColumnSelector(["pattern"])),

    ("encode", OneHotEncoder(sparse=False))
])

# Estimation pipeline :
baseline_pipeline = Pipeline([

    ("features", FeatureUnion([
        ("morpheme breakdown", morpheme_breakdown),
        ("pos encoder", pos_encoder),
        ("pattern encoder", pattern_encoder),
    ])),

    # Use an SVC classifier
    ("SVC", LinearSVC())
]
)

# Load corpus :
db: pd.DataFrame = load_corpus(CSV_PATH).fillna("").drop_duplicates()

# Remove words with root of len != 3
db = db[db.root.str.count(".") == 3]
corpus, roots = split_corpus_and_roots(db)

# Generate a train-test split
X_train, X_test, y_train, y_test = train_test_split(corpus, roots, test_size=0.33, random_state=42)

# Whole root prediction :
whole_root_score = np.average(
    cross_val_score(baseline_pipeline, corpus, roots, cv=10))
print(f"Cross val score for whole root : {whole_root_score}")

radical_models = []
preds = []
for radical_i in range(0, 3):
    # Evaluate cross val score of i'th radical model
    radical_i_cross_val_score = np.average(
        cross_val_score(baseline_pipeline, corpus, roots.apply(ith_radical_getter(radical_i)), cv=10))
    print(f"Cross val score for radical {radical_i + 1} : {radical_i_cross_val_score}")

    # Train model to predict i'th radical on train set
    model_i = baseline_pipeline.fit(X_train, y_train.apply(ith_radical_getter(radical_i)))
    radical_models.append(model_i)

    radical_i_test_prediction = model_i.predict(X_test)
    preds.append(radical_i_test_prediction)


def evaluate_combined_models(models: List[np.ndarray], y: pd.DataFrame):
    concat: np.ndarray = reduce(lambda x, y: x + y, models)
    return (concat == y).value_counts()[True] / len(y)


print(evaluate_combined_models(preds, y_test))
