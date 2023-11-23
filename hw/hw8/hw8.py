#!/usr/bin/env python3

"""
Benjamin Ye
CS/CNE/EE 156a: Learning Systems (Fall 2023)
November 20, 2023
Homework 8
"""

import pathlib
import sys

import numpy as np
import requests
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

CWD = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(CWD.parents[2]))
from cs156a import support_vector_machine

DATA_DIR = (CWD.parents[2] / "data").resolve()

if __name__ == "__main__":
    rng = np.random.default_rng()

    # Problems 2–4
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
    print(f"\n[HW8 P2–4]\nSVM with soft margins ({C=}) "
          f"and polynomial kernel ({Q=}):")
    clf = svm.SVC(C=C, kernel="poly", degree=Q, gamma=1, coef0=1)
    for digit in range(0, 10):
        x = data["train"][:, 1:]
        y = 2 * (data["train"][:, 0] == digit) - 1
        N_sv, E_out = support_vector_machine(
            x=x, y=y,
            x_test=data["test"][:, 1:], 
            y_test=2 * (data["test"][:, 0] == digit) - 1,
            clf=clf, rng=rng
        )
        E_in = 1 - clf.score(x, y)
        print(f"  {digit} vs. all: {N_sv=:,}, {E_in=:.4f}, {E_out=:.4f}")

    # Problems 5–6
    _x = data["train"][np.isin(data["train"][:, 0], (1, 5))]
    y = 2 * (_x[:, 0] == 1) - 1
    _x_test = data["test"][np.isin(data["test"][:, 0], (1, 5))]
    y_test = 2 * (_x_test[:, 0] == 1) - 1

    Cs = [0.0001, 0.001, 0.01, 0.1, 1]
    Qs = [2, 5]
    print("\n[HW8 P5–6]\nSVM with soft margins and polynomial kernel "
          "for 1 vs. 5 classifier:")
    for Q in Qs:
        for C in Cs:
            clf = svm.SVC(C=C, kernel="poly", degree=Q, gamma=1, coef0=1)
            N_sv, E_out = support_vector_machine(
                x=_x[:, 1:], y=y,
                x_test=_x_test[:, 1:], y_test=y_test,
                clf=clf, rng=rng
            )
            E_in = 1 - clf.score(_x[:, 1:], y)
            print(f"  {C=}, {Q=}: {N_sv=:,}, {E_in=:.4f}, {E_out=:.4f}")

    # Problems 7–8
    Q = 2
    n_fold = 10
    n_runs = 100

    clfs = [svm.SVC(C=C, kernel="poly", degree=Q, gamma=1, coef0=1) for C in Cs]
    ns_C = np.zeros_like(Cs, dtype=int)
    Es_cv_C = np.zeros_like(Cs, dtype=float)
    for _ in range(n_runs):
        Es_cv = np.fromiter((1 - cross_val_score(clf, _x[:, 1:], y, 
                                                 cv=n_fold).mean() 
                             for clf in clfs), dtype=float, count=len(clfs))
        ns_C[np.argmin(Es_cv)] += 1
        Es_cv_C += Es_cv
        _x, y = shuffle(_x, y)
    Es_cv_C /= n_runs
    best_C = np.argmax(ns_C)
    print("\n[HW8 P7–8]\nCross validation for SVM with soft margins "
        "and polynomial kernel (1 vs. 5 classifier):\n",
        f"  C={Cs[best_C]} is selected most often.\n",
        f"  E_cv={Es_cv_C[best_C]:.3f}",)
    
    # Problems 9–10
    print("\n[HW8 P9–10]\nSVM with soft margins and RBF kernel "
          "(1 vs. 5 classifier):")
    clfs = [svm.SVC(C=C, gamma=1) for C in [0.01, 1, 100, 1e4, 1e6]]
    for clf in [svm.SVC(C=C, gamma=1) for C in [0.01, 1, 100, 1e4, 1e6]]:
        _, E_out = support_vector_machine(
            x=_x[:, 1:], y=y, 
            x_test=_x_test[:, 1:], y_test=y_test,
            clf=clf, rng=rng
        )
        E_in = 1 - clf.score(_x[:, 1:], y)
        print(f"  C={clf.C:,g}, {E_in=:.4f}, {E_out=:.4f}")