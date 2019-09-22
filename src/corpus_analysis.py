from src.parser import load_corpus, CSV_PATH
import pandas as pd

corpus = pd.DataFrame(load_corpus(CSV_PATH).fillna("").drop_duplicates())

print("Size of corpus (number of distinct morpheme,root,pattern,pos combinations ) : {}".format(corpus.shape[0]))
for num_radicals in (2, 3, 4, 5):
    per_root_count = len(corpus[corpus.apply(lambda x: len(x.root) == num_radicals, axis=1)])
    print(fr"Number of roots of length {num_radicals} : {per_root_count} ")

corpus = corpus[corpus.root.str.count(".") == 3]
print("Size of corpus (number of distinct morpheme,root,pattern,pos combinations ) : {}".format(corpus.shape[0]))
print("Number of distinct word-root combinations {}".format(corpus[["morpheme", "root"]].drop_duplicates().shape[0]))

for category, entry_name in zip(("morphemes", "roots", "patterns", "POS"), ("morpheme", "root", "pattern", "pos")):
    unique_count = corpus[entry_name].unique().shape[0]
    print(f"Number of unique {category} : {unique_count}")



num_roots_per_morpheme = corpus[["morpheme", "root"]].drop_duplicates().groupby("morpheme").agg(
    count=pd.NamedAgg(column="morpheme", aggfunc='count'))
for num_roots in num_roots_per_morpheme["count"].unique():
    unique_count = len(num_roots_per_morpheme[num_roots_per_morpheme["count"] == num_roots])
    percent = int(unique_count / len(corpus["morpheme"].unique()) * 100)
    print(f"Number of morphemes with {num_roots} distinct roots : {unique_count}  ({percent}%)")
