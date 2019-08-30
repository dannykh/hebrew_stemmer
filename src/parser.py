import os
import xml.etree.ElementTree as ET
import pickle as pkl

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
    def __init__(self, morpheme, pos, score, root=None):
        self.root = root
        self.morpheme = morpheme
        self.pos = pos
        self.score = score


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


def get_words_from_article(morphologically_disambiguated_file_path):
    words = {}
    try:
        morphos = ET.parse(morphologically_disambiguated_file_path).getroot()
    except ET.ParseError as e:
        print("Could not parse {}".format(morphologically_disambiguated_file_path))
        return []

    for token in morphos.findall('.//token'):
        word = transliterate_string(token.attrib['surface'])
        for analysis in token.findall('.//'):
            if 'root' in analysis.attrib:
                root = transliterate_string(analysis.attrib['root'])
                if word not in words:
                    words[word] = {}
                if root not in words[word]:
                    words[word].update({root: 1})
                else:
                    words[word][root] += 1

    return words


def load_corpus_from_raw_files(dir_path='../raw_data'):
    words = {}
    for file_root, _, files in os.walk(dir_path):
        # if len(words) > CORPUS_MAX_SIZE:
        #     break
        for file_path in files:
            if '.xml' not in file_path:
                continue
            try:
                for word, roots in get_words_from_article(file_root + "/" + file_path).items():
                    if word not in words:
                        words[word] = roots
                    else:
                        for root in roots:
                            if root not in words[word]:
                                words[word].update({root: 1})
                            else:
                                words[word][root] += roots[root]
            except Exception as e:
                pass

    return words


def load_and_pickle_corpus(dir_path="../raw_data", dump_path=DUMP_PATH):
    corpus = load_corpus_from_raw_files(dir_path)
    with open(dump_path, 'wb') as fp:
        pkl.dump(corpus, fp)


def load_corpus(dump_path=DUMP_PATH):
    corpus = []
    with open(dump_path, 'rb') as fp:
        corpus = pkl.load(fp)
    return corpus


if __name__ == '__main__':
#     get_words_from_article('../raw_data/1.xml')
    # load_and_pickle_corpus()
    corpus = load_corpus()
    print("hey")
