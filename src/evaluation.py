from src.utils import load_split_corpus


def run_eval(trained_model, target_sets, wfp):
    for target_set_desc, target_set in target_sets:
        try:
            score = (trained_model.predict(target_set) == target_set.root).value_counts()[True] / len(
                target_set)
        except KeyError:
            score = 0
        wfp.write(f"Model score on {target_set_desc} test set : {score} \n")


def fit_and_eval(model, train, test_sets, wfp):
    y = train.root
    X = train.drop("root", axis=1)
    trained_model = model.fit(X, y)
    run_eval(trained_model, test_sets, wfp)


def run_test(model, model_name: str, test_proportion=0.2):
    # Load corpus :
    corpora = load_split_corpus(test_proportion=test_proportion)

    with open(f"../results/{model_name.replace(' ', '_')}.res", "w") as wfp:
        for train_corpus_name, train_corpus in corpora["train"].items():
            wfp.write(f" -- {model_name} trained on {train_corpus_name} : \n")
            fit_and_eval(model, train_corpus, corpora["test"].items(), wfp)
            wfp.write("\n")
