from itertools import product
from typing import List, Callable, T

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

max_word_len = 11

heb_alphabet = ["", 'a', 'b', 'g', 'd', 'h', 'w', 'z', 'x', 'v', 'i', 'k', 'l', 'm', 'n', 's', '&', 'p', 'c', 'q', 'r',
                '$', 't']
heb_letter_bigrams = sorted(product(heb_alphabet, heb_alphabet))
MAX_WORD_LENGTH = 20
preffixes = ['m', 'e', 'h', 'w', 'k', 'l', 'b', 'me', 'mh', 'ke', 'we', 'wh', 'wl', 'wke', 'lke', 'mke', 'mlke',
             'wlke']
patterns = ["", "Hif'il", "Hitpa'el", "Huf'al", "Nif'al", "Pa'al", "Pi'el", "Pu'al"]


def feature_bigrams_of_letters(word):
    # Yields a feature vector in which entry i = 1 iff the ith bigram of Hebrew letters (sorted alphabetically) appears
    # in the word
    feat_vec = [0] * len(heb_letter_bigrams)
    for i, big in enumerate(heb_letter_bigrams):
        feat_vec[i] = 1 if ''.join(big) in word.morpheme else 0

    return feat_vec


class StringDataSplit(TransformerMixin):
    def __init__(self, column: str, max_len: int, column_split_name=None):
        super(TransformerMixin, self)
        self.max_len = max_len
        self.column = column
        self.column_split_name = column_split_name if column_split_name is not None else column

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        return pd.DataFrame({f"{self.column_split_name}_{i}": X[self.column].str.split("", expand=True).get(i) for i in
                             range(1, self.max_len + 1)}).fillna("")


class ColumnSelector(TransformerMixin):
    def __init__(self, columns):
        super(TransformerMixin, self)
        self.columns = columns

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        return X[self.columns]


class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode

    def fit(self, X, y=None):
        return self  # not relevant here

    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class AlphabetEncoder:
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode

    def fit(self, X, y=None):
        return self  # not relevant here

    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        return X.apply(lambda col: [heb_alphabet.index(c) for c in col])

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class RowRemover:
    def __init__(self, criteria: Callable[[pd.DataFrame], pd.Series]):
        self.criteria = criteria

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.criteria(X)]

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X, y)


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
prefix_encoder = Pipeline([
    ("select prefix", ColumnSelector(["prefix"])),

    ("encode", OneHotEncoder(sparse=False))
])
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
