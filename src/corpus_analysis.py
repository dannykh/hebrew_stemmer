import pickle

from src.parser import load_corpus, CSV_PATH
import pandas as pd

"""
This module prints an analysis of the corpus and performs some data cleaning and splitting.
"""

corpus = pd.DataFrame(load_corpus(CSV_PATH).fillna("").drop_duplicates())

print("Size of corpus (number of distinct morpheme,root,pattern,pos combinations ) : {}".format(corpus.shape[0]))
for num_radicals in (2, 3, 4, 5):
    per_root_count = len(corpus[corpus.apply(lambda x: len(x.root) == num_radicals, axis=1)])
    print(fr"Number of roots of length {num_radicals} : {per_root_count} ")

corpus = corpus[corpus.root.str.count(".") == 3]
print("Size of corpus (number of distinct morpheme,root,pattern,pos combinations ) : {}".format(corpus.shape[0]))
print("Number of distinct word-root combinations {}".format(corpus[["morpheme", "root"]].drop_duplicates().shape[0]))

feature_names_to_columns = list(zip(("morphemes", "roots", "patterns", "POS"), ("morpheme", "root", "pattern", "pos")))
features = [feat for _, feat in feature_names_to_columns]

for category, entry_name in feature_names_to_columns:
    unique_count = corpus[entry_name].unique().shape[0]
    print(f"Number of unique {category} : {unique_count}")

# Remove all data with categories which are too rare (have less than a predefined frequency)
for feature in ["pos", "pattern"]:
    print(f"{feature} category frequency : ")
    freq = corpus.groupby(feature).count().morpheme / len(corpus)
    print((freq * 100).to_string())
    low_freq = freq[freq < 0.2].dropna().index
    corpus = corpus[~corpus[feature].isin(low_freq)]

num_roots_per_morpheme = corpus[["morpheme", "root"]].drop_duplicates().groupby("morpheme").agg(
    count=pd.NamedAgg(column="morpheme", aggfunc='count'))
for num_roots in num_roots_per_morpheme["count"].unique():
    unique_count = len(num_roots_per_morpheme[num_roots_per_morpheme["count"] == num_roots])
    percent = int(unique_count / len(corpus["morpheme"].unique()) * 100)
    print(f"Number of morphemes with {num_roots} distinct roots : {unique_count}  ({percent}%)")

special_radicals_map = {
    1: ("i", "w", "n"),
    2: ("i", "w", "h"),
    3: ("h", "i")
}
special_roots_df_lst = []
for rad_i, special_vals in special_radicals_map.items():
    for ch in special_vals:
        ri_special = corpus[corpus.root.str[rad_i - 1] == ch]
        special_roots_df_lst.append(ri_special)
        ri_special_count = len(ri_special)
        percent = int(ri_special_count / len(corpus) * 100)
        print(f"Number of roots with R{rad_i} = {ch} : {ri_special_count}, ({percent}%)")

r2_r3_same = corpus[corpus.root.str[1] == corpus.root.str[2]]
special_roots_df_lst.append(r2_r3_same)
r2_r3_same_count = len(r2_r3_same)
percent = int(r2_r3_same_count / len(corpus) * 100)
print(f"Number of roots with R2 = R3 : {r2_r3_same_count}, ({percent}%)")

irregular = pd.concat(special_roots_df_lst).drop_duplicates(keep="first")
regular = pd.concat([irregular, corpus]).drop_duplicates(keep=False)

with open("../data/corpus_divided_reg_irreg.pkl", "wb") as fp:
    pickle.dump((regular, irregular), fp)

print(
    f"Total words with irregular root : {len(irregular)} ({int(len(irregular) / len(corpus) * 100)}%)")
print(f"Total words with regular root : {len(regular)} ({int(len(regular) / len(corpus) * 100)}%)")

