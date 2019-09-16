from functools import reduce
from itertools import product
from typing import Dict

from sklearn.tree import DecisionTreeClassifier
import numpy as np
from src.parser import Word, load_corpus_from_raw_files
from sklearn.model_selection import cross_val_score
import pprint
from src.parser import load_corpus

"""
A word's stem retrieval shall yield a list of tuples : (<root>, <score>) - a root and it's score as the word's stem.
The input shall be an object implementing get_morpheme().

"""

# TODO : Binary features maybe useless, change to categorical


heb_alphabet = [
    'a', 'b', 'g', 'd', 'h', 'w', 'z', 'x', 'v', 'i', 'k', 'l', 'm', 'n', 's', 'y', 'p', 'c', 'q', 'r', 'e', 't']

# An alphabetically sorted list of Hebrew letter bigrams:
heb_letter_bigrams = sorted(product(heb_alphabet, heb_alphabet))

MAX_WORD_LENGTH = 20

preffixes = ['m', 'e', 'h', 'w', 'k', 'l', 'b', 'me', 'mh', 'ke', 'we', 'wh', 'wl', 'wke', 'lke', 'mke', 'mlke',
             'wlke']

patterns = ["", "Hif'il", "Hitpa'el", "Huf'al", "Nif'al", "Pa'al", "Pi'el", "Pu'al"]

pattern_matchers = ['^(...)ti$', '^(...)t$', '$(...)^', '^(...)h$', '^(...)nw$', '^(...)tm$', '^(...)tn$', '^(...)w$',
                    '^a(...)$',
                    '^t(...)$', '^t(...)i$', '^i(...)$', '^n(...)$', '^t(...)w$', '^t(...)nh$', '^i(...)w$', '^(...)i$',
                    '^(...)nh$',
                    '^n(...)ti$', '^n(...)t$', '^n(...)$', '^n(...)h$', '^n(...)nw$', '^n(...)tm$', '^n(...)tn$',
                    '^n(...)w$', '^t(...)i$',
                    '^t(...)w$', '^n(...)im$', '^n(...)wt$', '^h(...)$', '^h(...)i$', '^h(...)w$', '^h(...)nh$',
                    '^lh(...)$', '^h(...)$',
                    '^h(...)ti$', '^h(...)t$', '^h..i.$', '^h..i.h$', '^h(...)nw$', '^h(...)tm$', '^h(...)tn$',
                    '^h..i.w$', '^a..i.$',
                    '^t..i.$', '^t..i.i$', '^i..i.$', '^n..i.$', '^t..i.w$', '^t..i.nh$', '^i..i.w$', '^m..i.$',
                    '^m..i.h$', '^m..i.im$',
                    '^m..i.wt$', '^h(...)$', '^h..i.i$', '^h..i.w$', '^h(...)nh$', '^lh..i.$', '^h(...)h$', '^h(...)w$',
                    '^m(...)$',
                    '^m(...)t$', '^m(...)im$', '^m(...)wt$', '^l(...)$', '^ht(...)ti$', '^ht(...)t$', '^ht(...)$',
                    '^ht(...)h$',
                    '^ht(...)nw$', '^at(...)$', '^tt(...)$', '^tt(...)i$', '^it(...)$', '^nt(...)$', '^tt(...)w$',
                    '^tt(...)nh$',
                    '^it(...)w$', '^mt(...)$', '^mt(...)t$', '^mt(...)im$', '^mt(...)wt$', '^ht(...)i$', '^ht(...)nh$',
                    '^lht(...)$']


def feature_locations_of_letters(word: Word):
    # Limiting word length to 20, we note feature i*L ( 0 < i <= 20, 1<= L <= 22 ) as " the ith letter is letter #L.
    # This yields a feature vector of size 20*22 = 440
    feat_vec = [0] * len(heb_alphabet) * MAX_WORD_LENGTH
    for i, lett in enumerate(word.morpheme, 1):
        feat_vec[(heb_alphabet.index(lett) + 1) * i] = 1

    return feat_vec[1:]  # Discarding 0th place as it is irrelevant


def feature_bigrams_of_letters(word):
    # Yields a feature vector in which entry i = 1 iff the ith bigram of Hebrew letters (sorted alphabetically) appears
    # in the word
    feat_vec = [0] * len(heb_letter_bigrams)
    for i, big in enumerate(heb_letter_bigrams):
        feat_vec[i] = 1 if ''.join(big) in word.morpheme else 0

    return feat_vec


def feature_pattern(word):
    feat_vec = [0] * len(patterns)
    feat_vec[patterns.index(word.pattern)] = 1
    return feat_vec


def feature_prefixes(word):
    return [1 if word.morpheme.startswith(x) else 0 for x in preffixes]


def feature_root(word):
    return word.root


feature_extractors = [feature_bigrams_of_letters, feature_pattern]


def word_to_feature_vec(word):
    feat_vec = []
    for feat_ext in feature_extractors:
        feat_vec += feat_ext(word)

    return feat_vec


def corpus_to_feature_vectors(corpus):
    return np.array([word_to_feature_vec(word) for word in corpus])


def get_word_tags(corpus):
    return np.array([word.root for word in corpus])


def clean_corpus(corpus):
    # Remove multiple occurrences of word. Keep highest probability
    pass


if __name__ == "__main__":
    # corpus = load_corpus_from_raw_files(r"..\data\haaretz_tagged_xmlFiles")
    corpus = sorted(load_corpus(r"..\data\corpus.pkl"), key=lambda word: word.morpheme)
    clf = DecisionTreeClassifier()
    X = corpus_to_feature_vectors(corpus)
    Y = get_word_tags(corpus)
    # clf.fit(X, Y)
    # print(clf.predict([word_to_feature_vec(Word('iarx', 'verb', 1.0))]))
    print(np.average(cross_val_score(clf, X, Y, cv=10)))
