import os
import xml.etree.ElementTree as ET
import pickle as pkl
from typing import Dict, List
from functools import reduce

DUMP_PATH = "../data/corpus.pkl"

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
        "ע": "y",
        "פ": "p",
        "ף": "p",
        "צ": "c",
        "ץ": "c",
        "ק": "q",
        "ר": "r",
        "ש": "e",
        "ת": "t"
    }

    return ''.join([heb_char_trns[heb_chr] if heb_chr in heb_char_trns else heb_chr for heb_chr in heb_str])


class Word:
    def __init__(self, morpheme: str, pos: str, pattern: str = None, root: str = None):
        self.root = root
        self.morpheme = morpheme
        self.pos = pos
        self.pattern = pattern
        self.count = 0
        self.score = 0.0

    def __eq__(self, other: "Word"):
        return self.morpheme == other.morpheme and self.root == other.root and self.pos == other.pos \
               and self.pattern == other.pattern

    def __repr__(self):
        return "{},{},{},{}".format(self.morpheme, self.root, self.pos, self.pattern)

    def __hash__(self):
        return hash("".join((self.root, self.morpheme, self.pos, self.pattern)))

    def __str__(self):
        return self.__repr__()


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


def get_words_from_article(morphologically_disambiguated_file_path: str) -> List[Word]:
    words = []
    try:
        morphos = ET.parse(morphologically_disambiguated_file_path).getroot()
    except ET.ParseError as e:
        print("Could not parse {}".format(morphologically_disambiguated_file_path))
        return []

    # Iterate words in text
    for token in morphos.findall('.//token'):
        morpheme = transliterate_string(token.attrib['surface'])
        # Iterate possible word analysis
        for analysis in token.findall('analysis'):
            # skip analysis with 0 score
            if "score" not in analysis.attrib or float(analysis.attrib["score"]) == 0:
                continue
            for base in analysis.findall("base"):
                for pos in base.findall(".//"):
                    word = Word(morpheme, pos.tag, pos.attrib["binyan"] if "binyan" in pos.attrib else "",
                                transliterate_string(pos.attrib["root"]) if "root" in pos.attrib else None)
                    # Skip if no root
                    if word.root is None:
                        continue
                    words += [word]
                    # if morpheme not in words:
                    #     words[morpheme] = [word]
                    # else:
                    #     if word not in words[morpheme]:
                    #         words[morpheme].update({word: 1})
                    #     else:
                    #         words[morpheme][word] += 1

    return words


def load_corpus_from_raw_files(dir_path: str) -> List[Word]:
    corpus_words = []
    for file_root, _, files in os.walk(dir_path):
        # if len(words) > CORPUS_MAX_SIZE:
        #     break
        for file_path in files:
            if '.xml' not in file_path:
                continue
            try:
                corpus_words += get_words_from_article(file_root + "/" + file_path)
                # for morph, words in get_words_from_article(file_root + "/" + file_path).items():
                #     if morph not in corpus_words:
                #         corpus_words[morph] = words
                #     else:
                #         for word, count in words.items():
                #             if word not in corpus_words[morph]:
                #                 corpus_words[morph].update({word: count})
                #             else:
                #                 corpus_words[morph][word] += count
            except IndentationError as e:
                print("Could not parse {}. {}".format(file_path, e))

    return corpus_words


def load_and_pickle_corpus(dir_path, dump_path):
    corpus = load_corpus_from_raw_files(dir_path)
    with open(dump_path, 'wb') as fp:
        pkl.dump(corpus, fp)


def load_corpus(dump_path):
    with open(dump_path, 'rb') as fp:
        corpus = pkl.load(fp)
    return corpus


if __name__ == '__main__':
    print([transliterate_string(x).replace(".","(.)") for x in
           ["^...תי$", "^...ת$", "$...^", "^...ה$", "^...נו$", "^...תם$", "^...תן$", "^...ו$",
            "^א...$", "^ת...$", "^ת...י$", "^י...$", "^נ...$", "^ת...ו$", "^ת...נה$", "^י...ו$",
            "^...י$", "^...נה$", "^נ...תי$", "^נ...ת$", "^נ...$", "^נ...ה$", "^נ...נו$",
            "^נ...תם$", "^נ...תן$", "^נ...ו$", "^ת...י$", "^ת...ו$", "^נ...ים$", "^נ...ות$",
            "^ה...$", "^ה...י$", "^ה...ו$", "^ה...נה$", "^לה...$", "^ה...$", "^ה...תי$",
            "^ה...ת$", "^ה..י.$", "^ה..י.ה$", "^ה...נו$", "^ה...תם$", "^ה...תן$", "^ה..י.ו$",
            "^א..י.$", "^ת..י.$", "^ת..י.י$", "^י..י.$", "^נ..י.$", "^ת..י.ו$", "^ת..י.נה$",
            "^י..י.ו$", "^מ..י.$", "^מ..י.ה$", "^מ..י.ים$", "^מ..י.ות$", "^ה...$", "^ה..י.י$",
            "^ה..י.ו$", "^ה...נה$", "^לה..י.$", "^ה...ה$", "^ה...ו$", "^מ...$", "^מ...ת$",
            "^מ...ים$", "^מ...ות$", "^ל...$", "^הת...תי$", "^הת...ת$", "^הת...$", "^הת...ה$",
            "^הת...נו$", "^את...$", "^תת...$", "^תת...י$", "^ית...$", "^נת...$", "^תת...ו$",
            "^תת...נה$", "^ית...ו$", "^מת...$", "^מת...ת$", "^מת...ים$", "^מת...ות$", "^הת...י$",
            "^הת...נה$", "^להת...$"]])
    corpus = load_corpus_from_raw_files(r"..\data\haaretz_tagged_xmlFiles")
    load_and_pickle_corpus(r"..\data\haaretz_tagged_xmlFiles", r"..\data\corpus.pkl")
    # get_words_from_article(r"..\data\haaretz_tagged_xmlFiles\1.xml")
    # load_and_pickle_corpus()
    # corpus = load_corpus()
    # print("hey")
