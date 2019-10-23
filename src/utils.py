import pickle

import pandas as pd
from sklearn.model_selection import train_test_split


def extract_patterns(corpus):
    return sorted(list(set([word.pattern for word in corpus if word.pattern])))


def ith_radical_getter(radical_i):
    return lambda x: x[radical_i] if len(x) > radical_i else ""


def load_split_corpus(test_proportion=0.2):
    with open("../data/corpus_divided_reg_irreg.pkl", "rb") as fp:
        regular, irregular = pickle.load(fp)

    regular_train, regular_test = train_test_split(regular, test_size=test_proportion)
    irregular_train, irregular_test = train_test_split(irregular, test_size=test_proportion)

    combined_train = pd.concat((regular_train, irregular_train))
    combined_test = pd.concat((regular_test, irregular_test))

    return {
        "test": {
            "regular": regular_test,
            "irregular": irregular_test,
            "combined": combined_test
        },
        "train": {
            "regular": regular_train,
            "irregular": irregular_train,
            "combined": combined_train
        },
    }
