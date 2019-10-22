from pprint import pprint

from src.parser import load_corpus


def extract_patterns(corpus):
    return sorted(list(set([word.pattern for word in corpus if word.pattern])))


if __name__ == '__main__':
    pprint(extract_patterns(load_corpus(r"..\data\corpus.pkl")))


def ith_radical_getter(radical_i):
    return lambda x: x[radical_i] if len(x) > radical_i else ""
