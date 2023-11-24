#!/usr/bin/env python3

"""
Benjamin Ye
CS/CNE/EE 156a: Learning Systems (Fall 2023)
November 13, 2023
Homework 7
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests
from sklearn import svm

CWD = Path(__file__)
sys.path.insert(0, str(CWD.parents[2]))
from cs156a import (LinearRegression, Perceptron, target_function_random_line,
                    generate_data, validate_binary)

DATA_DIR = CWD.parents[2] / "data"

if __name__ == "__main__":
    rng = np.random.default_rng()

    ### Problems 1–5 ##########################################################

    DATA_DIR.mkdir(exist_ok=True)
    raw_data = {}
    for prefix in ["in", "out"]:
        if not (DATA_DIR / f"{prefix}.dta").exists():
            r = requests.get(f"http://work.caltech.edu/data/{prefix}.dta")
            with open(DATA_DIR / f"{prefix}.dta", "wb") as f:
                f.write(r.content)
        raw_data[prefix] = np.loadtxt(DATA_DIR / f"{prefix}.dta")

    ns = (25, len(raw_data["in"]) - 25)
    data = np.array_split(raw_data["in"], (ns[0],))
    transforms = (
        lambda x: np.ones((len(x), 1), dtype=float), 
        lambda x: x,
        lambda x: x[:, :1] ** 2, 
        lambda x: x[:, 1:] ** 2, 
        lambda x: np.prod(x, axis=1, keepdims=True), 
        lambda x: np.abs(x[:, :1] - x[:, 1:]), 
        lambda x: np.abs(x[:, :1] + x[:, 1:])
    )
    reg = LinearRegression(
        vf=validate_binary, 
        transform=lambda x: np.hstack(tuple(f(x) for f in transforms[:k])),
        rng=rng
    )
    df = pd.DataFrame(columns=["split", "k", "training error", 
                               "validation error", "out-of-sample error"])
    for i in range(2):
        for k in np.arange(3, 8):
            E_train = reg.train(data[i][:, :-1], data[i][:, -1])
            E_validate = reg.get_error(data[1 - i][:, :-1], data[1 - i][:, -1])
            E_out = reg.get_error(raw_data["out"][:, :-1], 
                                  raw_data["out"][:, -1])
            df.loc[len(df)] = (f"{ns[i]}:{ns[1 - i]}", k, 
                               E_train, E_validate, E_out)
    print("\n[Homework 7 Problems 1–5]\n"
          "Linear regression with nonlinear transformation:\n",
          df.to_string(index=False), sep="")

    ### Problem 6 #############################################################

    x = rng.uniform(size=(10_000_000, 2))
    e_1, e_2 = x.mean(axis=0)
    e = x.min(axis=1).mean()
    print("\n[Homework 7 Problem 6]\n"
          "The expected values for paired independent uniform random "
          f"variables and their minimum are {e_1:.6f}, "
          f"{e_2:.6f}, and {e:.6f}, respectively.")

    ### Problems 8–10 ########################################################

    N_runs = 1_000
    f = target_function_random_line(rng=rng)
    pla = Perceptron(vf=validate_binary)
    clf = svm.SVC(C=np.finfo(float).max, kernel="linear")
    df = pd.DataFrame(columns=["N", "SVM > perceptron",
                               "number of support vectors"])
    for N_train in (10, 100):
        N_test = 99 * N_train
        counters = np.zeros(2, dtype=float)
        for _ in range(N_runs):
            while True:
                x_train, y_train = generate_data(N_train, f, bias=True, 
                                                 rng=rng)
                if not np.allclose(y_train, y_train[0]):
                    break
            x_test, y_test = generate_data(N_test, f, bias=True, rng=rng)
            pla.train(x_train, y_train)
            clf.fit(x_train[:, 1:], y_train)
            counters += (
                1 - clf.score(x_test[:, 1:], y_test) 
                    < pla.get_error(x_test, y_test),
                clf.n_support_.sum()
            )
        counters /= N_runs
        df.loc[len(df)] = N_train, 100 * counters[0], counters[1]
    print("\n[Homework 7 Problems 8–10]\n"
          "Comparison of perceptron and support vector machine (SVM):\n",
          df.to_string(index=False), sep="")