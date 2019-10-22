import os
import pprint
import xml.etree.ElementTree as ET
import pickle as pkl
from itertools import count
from typing import Dict, List, Callable, Optional, Set
from functools import reduce
import pandas as pd

CSV_PATH = "../data/corpus.csv"

CORPUS_MAX_SIZE = 10000  # words


def transliterate_string(heb_str):
    heb_char_trns = {
        "א": "a",
        "ב": "b",
        "ג": "g",
        "ד": "d",
        "ה": "h",
        "ו": "w",
        "ז": "z",
        "ח": "x",
        "ט": "v",
        "י": "i",
        "כ": "k",
        "ך": "k",
        "ל": "l",
        "מ": "m",
        "ם": "m",
        "נ": "n",
        "ן": "n",
        "ס": "s",
        "ע": "&",
        "פ": "p",
        "ף": "p",
        "צ": "c",
        "ץ": "c",
        "ק": "q",
        "ר": "r",
        "ש": "$",
        "ת": "t"
    }

    return ''.join([heb_char_trns[heb_chr] if heb_chr in heb_char_trns else heb_chr for heb_chr in heb_str])


"""
# Obsolete ! remnants of older implementation, kept for historical educational documentation
class Word:
    def __init__(self, morpheme: str, pos: str, pattern: str = None, root: str = None):
        self.root = root
        self.morpheme = morpheme
        self.pos = pos
        self.pattern = pattern

    def __eq__(self, other: "Word"):
        return self.morpheme == other.morpheme and self.root == other.root and self.pos == other.pos \
               and self.pattern == other.pattern

    def __repr__(self):
        return "{},{},{},{}".format(self.morpheme, self.root, self.pos, self.pattern)

    def __hash__(self):
        return hash("".join((self.root, self.morpheme, self.pos, self.pattern)))

    def __str__(self):
        return self.__repr__()


class WordInCorpus(Word):
    def __init__(self, word: Word):
        super(Word, self).__init__(word.morpheme, word.pos, word.pattern, word.root)
        self.count = 0


class Corpus(object):
    def __init__(self, init_words: Set[Word] = None):
        self.words = set() if init_words is None else {WordInCorpus(word) for word in init_words}

    def add_word(self, word: Word, count: int = 1):
        self.words = self.words.get(word, 0) + count

    def get_words(self, order_method: Callable[[List[Word]], List[Word]] = lambda lst: lst,
                  **kwargs) -> List[Word]:
        return list(order_method(list(self.words.keys())))

    def get_word_counts(self):
        return self.words

    def __contains__(self, item: Word):
        return type(item) is Word and item in self.words

    def __getitem__(self, item: Word):
        return self.words[item]

    def as_dataframe(self):
        columns = ["morpheme", "root", "pos", "pattern", "count"]
        val_dct = [{column: getattr(word, column) for column in columns} for word in self.words]
        return pd.DataFrame(val_dct, columns=columns)


def parse_roots(file_name):
    with open("roots" + file_name + ".txt", "w", encoding="utf-8") as root_file:
        tree = ET.parse('../data/' + file_name + '.xml')
        corpus = tree.getroot()

        for token in corpus.findall('.//token'):
            word = token.attrib["surface"]
            for analysis in token:
                if analysis.attrib["score"] == "1.0":
                    for item in analysis:
                        if item.tag == "base" and "root" in item[0].attrib:
                            root_file.write(word + " " + item[0].attrib["root"] + "\n")

"""


def get_words_from_article(morphologically_disambiguated_file_path: str) -> List[Dict[str, str]]:
    words = []
    try:
        morphos = ET.parse(morphologically_disambiguated_file_path).getroot()
    except ET.ParseError as e:
        print("Could not parse {}".format(morphologically_disambiguated_file_path))
        return []

    # Iterate words in text
    for token in morphos.findall('.//token'):
        morpheme = transliterate_string(token.attrib['surface']).strip()
        # Iterate possible word analysis
        for analysis in token.findall('analysis'):
            # skip analysis with 0 score
            # if "score" not in analysis.attrib or float(analysis.attrib["score"]) == 0:
            #     continue
            prefix = analysis.find("prefix").attrib["surface"] if analysis.find("prefix") is not None else ""
            for base in analysis.findall("base"):
                for pos in base.findall(".//"):
                    if "root" not in pos.attrib:
                        continue
                    words.append({"morpheme": morpheme,
                                  "pos": pos.tag.strip(),
                                  "pattern": pos.get("binyan", "").strip(),
                                  "prefix": transliterate_string(prefix),
                                  "root": transliterate_string(pos.attrib["root"].strip())})

    return words


def load_corpus_from_raw_files(dir_path: str) -> pd.DataFrame:
    words = []
    for file_root, _, files in os.walk(dir_path):
        # if len(words) > CORPUS_MAX_SIZE:
        #     break
        for file_path in files:
            if '.xml' not in file_path:
                continue
            try:
                words += get_words_from_article(file_root + "/" + file_path)
            except Exception as e:
                print("Could not parse {}. {}".format(file_path, e))

    return pd.DataFrame(words, columns=["morpheme", "pos", "pattern", "prefix", "root"])


def load_corpus_to_csv(dir_path: str, csv_path: str):
    corpus_df = load_corpus_from_raw_files(dir_path)
    corpus_df.to_csv(csv_path)


def load_corpus(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, index_col=0)


def split_corpus_and_roots(corpus: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    roots = corpus.root
    return corpus.drop("root", 1), roots


if __name__ == '__main__':
    load_corpus_to_csv(r"..\data\haaretz_tagged_xmlFiles", CSV_PATH)
    corpus = load_corpus(CSV_PATH)
    pprint.pprint(corpus)
