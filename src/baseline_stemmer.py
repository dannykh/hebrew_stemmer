from itertools import product
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from src.parser import Word
from sklearn.model_selection import cross_val_score

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


def feature_locations_of_letters(word):
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


def feature_prefixes(word):
    return [1 if word.morpheme.startswith(x) else 0 for x in preffixes]


feature_extractors = [feature_locations_of_letters, feature_prefixes]


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
    clf = DecisionTreeClassifier()
    corpus = load_corpus()
    X = corpus_to_feature_vectors(corpus)
    Y = get_word_tags(corpus)
    clf.fit(X, Y)
    # print(clf.predict([word_to_feature_vec(Word('iarx', 'verb', 1.0))]))
    print(cross_val_score(clf, X, Y, cv=10))
