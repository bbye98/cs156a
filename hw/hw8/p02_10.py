from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

DATA_DIR = Path(__file__).parents[2] / "data"

if __name__ == "__main__":
    rng = np.random.default_rng()

    DATA_DIR.mkdir(exist_ok=True)
    data = {}
    for dataset in ["train", "test"]:
        file = f"features.{dataset}"
        if not (DATA_DIR / file).exists():
            r = requests.get(f"http://www.amlbook.com/data/zip/{file}")
            with open(DATA_DIR / file, "wb") as f:
                f.write(r.content)
        data[dataset] = np.loadtxt(DATA_DIR / file)

    C = 0.01
    Q = 2
    clf = svm.SVC(C=C, kernel="poly", degree=Q, gamma=1, coef0=1)
    df = pd.DataFrame(columns=["classifier", "number of support vectors", 
                               "in-sample error", "out-of-sample error"])
    for digit in range(10):
        x_train = data["train"][:, 1:]
        y_train = 2 * (data["train"][:, 0] == digit) - 1
        clf.fit(x_train, y_train)
        df.loc[digit] = (
            f"{digit} vs. all", 
            clf.n_support_.sum(),
            1 - clf.score(x_train, y_train),
            1 - clf.score(data["test"][:, 1:], 
                          2 * (data["test"][:, 0] == digit) - 1)
        )
    print("\n[Homework 8 Problems 2–4]\n"
          f"Soft margin ({C=}) SVM with polynomial kernel ({Q=}):\n",
          df.to_string(index=False), sep="")

    x_train = data["train"][np.isin(data["train"][:, 0], (1, 5))]
    y_train = 2 * (x_train[:, 0] == 1) - 1
    x_test = data["test"][np.isin(data["test"][:, 0], (1, 5))]
    y_test = 2 * (x_test[:, 0] == 1) - 1
    df = pd.DataFrame(columns=["C", "Q", "number of support vectors",
                               "in-sample error", "out-of-sample error"])
    for Q in (2, 5):
        for C in (Cs := (0.0001, 0.001, 0.01, 0.1, 1)):
            clf = svm.SVC(C=C, kernel="poly", degree=Q, gamma=1, coef0=1)
            clf.fit(x_train[:, 1:], y_train)
            df.loc[len(df)] = (
                C, Q, clf.n_support_.sum(),
                1 - clf.score(x_train[:, 1:], y_train),
                1 - clf.score(x_test[:, 1:], y_test)
            )
    print("\n[Homework 8 Problems 5–6]\n"
          f"Soft margin ({C=}) SVM with polynomial kernel ({Q=}) for "
          "1 vs. 5 classifier:\n",
          df.to_string(index=False), sep="")

    Q = 2
    N_runs = 100
    N_folds = 10
    clfs = [svm.SVC(C=C, kernel="poly", degree=Q, gamma=1, coef0=1) 
            for C in Cs]
    counters = np.zeros((2, len(Cs)), dtype=float)
    for _ in range(N_runs):
        Es_cv = tuple(1 - cross_val_score(clf, x_train[:, 1:], y_train, 
                                          cv=N_folds).mean()
                      for clf in clfs)
        counters[0] += Es_cv
        counters[1, np.argmin(Es_cv)] += 1
        x_train, y_train = shuffle(x_train, y_train)
    counters /= N_runs
    df = pd.DataFrame({"C": Cs, "cross-validation error": counters[0],
                       "selection rate": counters[1]})
    print("\n[Homework 8 Problems 7–8]\n"
          f"Cross-validation error for soft margin ({C=}) SVM with "
          f"polynomial kernel ({Q=}) for 1 vs. 5 classifier:\n",
          df.to_string(index=False), sep="")

    df = pd.DataFrame(columns=["C", "in-sample error", "out-of-sample error"])
    for C in (0.01, 1, 100, 1e4, 1e6):
        clf = svm.SVC(C=C, gamma=1)
        clf.fit(x_train[:, 1:], y_train)
        df.loc[len(df)] = (
            clf.C, 
            1 - clf.score(x_train[:, 1:], y_train),
            1 - clf.score(x_test[:, 1:], y_test)
        )
    print("\n[Homework 8 Problems 9–10]\n"
          "Soft margin SVM with RBF kernel for 1 vs. 5 classifier:\n",
          df.to_string(index=False), sep="")