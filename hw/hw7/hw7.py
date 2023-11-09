#!/usr/bin/env python3

"""
Benjamin Ye
CS/CNE/EE 156a: Learning Systems (Fall 2023)
November 13, 2023
Homework 7
"""

import pathlib
import sys

import numpy as np
import requests
from sklearn import svm

CWD = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CWD.parent))
from cs156a import (
    linear_regression, validate_binary, target_function_random_line, 
    generate_data, perceptron, support_vector_machine
)

DATA_DIR = (CWD / "../data").resolve()

if __name__ == "__main__":
    rng = np.random.default_rng()

    # Problems 1–5
    DATA_DIR.mkdir(exist_ok=True)
    raw_data = {}
    for prefix in ["in", "out"]:
        if not (DATA_DIR / f"{prefix}.dta").exists():
            r = requests.get(f"http://work.caltech.edu/data/{prefix}.dta")
            with open(DATA_DIR / f"{prefix}.dta", "wb") as f:
                f.write(r.content)
        raw_data[prefix] = np.loadtxt(DATA_DIR / f"{prefix}.dta")

    print("\n[HW7 P1–5]")
    ns = (25, len(raw_data["in"]) - 25)
    data = np.array_split(raw_data["in"], (ns[0],))
    transform_funcs = (
        lambda x: np.ones((len(x), 1), dtype=float), 
        lambda x: x,
        lambda x: x[:, :1] ** 2, 
        lambda x: x[:, 1:] ** 2, 
        lambda x: np.prod(x, axis=1, keepdims=True), 
        lambda x: np.abs(x[:, :1] - x[:, 1:]), 
        lambda x: np.abs(x[:, :1] + x[:, 1:])
    )
    for i in range(2):
        print(f"Linear regression statistics for {ns[i]}:{ns[1 - i]} split:")
        for k in np.arange(3, 8):
            w, E_in, E_out = linear_regression(
                vf=validate_binary, 
                x=data[i][:, :-1],
                y=data[i][:, -1],
                transform=lambda x: np.hstack(
                    tuple(f(x) for f in transform_funcs[:k])
                ),
                x_test=raw_data["out"][:, :-1], 
                y_test=raw_data["out"][:, -1], 
                x_validate=data[1 - i][:, :-1],
                y_validate=data[1 - i][:, -1],
                hyp=True
            )
            print(f"  {k=}, E_in_test={E_in[0]:.3f}, "
                  f"E_in_validate={E_in[1]:.3f}, {E_out=:.3f}")

    # Problem 6
    x = rng.uniform(size=(10_000_000, 2))
    e_1, e_2 = x.mean(axis=0)
    e = x.min(axis=1).mean()
    print("\n[HW7 P6]\nExpected values for continuous uniform distribution:",
        f"  {e_1=:.3f}, {e_2=:.3f}, {e=:.3f}", sep="\n")

    # Problems 8–10
    Ns = (10, 100)
    N_runs = 1_000
    N_test = 100_000

    print(f"\n[HW7 P8–10]\nPLA vs. SVM with hard margins over {N_runs:,} runs:")
    f = target_function_random_line(rng=rng)
    clf = svm.SVC(C=np.finfo(float).max, kernel="linear")
    for N in Ns:
        prob_svm = 0
        N_sv_avg = 0
        for _ in range(N_runs):
            while True:
                x, y = generate_data(N, f, bias=True, rng=rng)
                if not np.allclose(y, y[0]):
                    break
            x_test, y_test = generate_data(N_test, f, bias=True, rng=rng)
            _, E_out_pla = perceptron(N, f, vf=validate_binary, x=x, y=y, 
                                    x_test=x_test, y_test=y_test, rng=rng)
            N_sv, E_out_svm = support_vector_machine(
                N, f, vf=validate_binary, x=x, y=y, x_test=x_test, y_test=y_test,
                clf=clf, rng=rng
            )
            prob_svm += E_out_svm < E_out_pla
            N_sv_avg += N_sv
        prob_svm /= N_runs
        N_sv_avg /= N_runs
        print(f"  {N=}, {prob_svm=:.3f}, {N_sv_avg=:.3f}")