
import pandas as pd

from src.parser import load_corpus, CSV_PATH

if __name__ == '__main__':
    corpus : pd.DataFrame = load_corpus(CSV_PATH)
    morpheme_split = corpus.morpheme.str.extractall("(.)").unstack()
    root_split = corpus.root.str.extractall("(.)").unstack()

    joined = pd.concat([morpheme_split,root_split], axis=1)
    print(joined)