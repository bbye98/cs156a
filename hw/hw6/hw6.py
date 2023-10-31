#!/usr/bin/env python3

"""
Benjamin Ye
CS/CNE/EE 156a: Learning Systems (Fall 2023)
November 6, 2023
Homework 6
"""

import pathlib
import sys

import numpy as np
import requests

CWD = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CWD.parent))
from cs156a import linear_regression, validate_binary

if __name__ == "__main__":

    (CWD / "data").mkdir(exist_ok=True)
    data = {"train": "in.dta", "test": "out.dta"}
    for dataset, file in data.items():
        if not (CWD / "data" / file).exists():
            r = requests.get(f"http://work.caltech.edu/data/{file}")
            with open(CWD / "data" / file, "wb") as f:
                f.write(r.content)
        data[dataset] = np.loadtxt(CWD / "data" / file)

    transform = lambda x: np.hstack((
        np.ones((len(x), 1), dtype=float), 
        x, 
        x[:, :1] ** 2, 
        x[:, 1:] ** 2, 
        x[:, :1] * x[:, 1:], 
        np.abs(x[:, :1] - x[:, 1:]), 
        np.abs(x[:, :1] + x[:, 1:])
    ))

    print("\n[HW6 P2–6]\nLinear regression (without regularization) "
        "statistics:")
    E_in, E_out = linear_regression(
        vf=validate_binary, x=data["train"][:, :-1], y=data["train"][:, -1],
        transform=transform, x_test=data["test"][:, :-1], 
        y_test=data["test"][:, -1]
    )
    print(f"  {E_in=:.3f}, {E_out=:.3f}")
    print("Linear regression (with weight decay regularization using "
          "lambda=10^k) statistics:")
    for k in (ks := np.arange(-5, 7)):
        E_in, E_out = linear_regression(
            vf=validate_binary, x=data["train"][:, :-1], y=data["train"][:, -1],
            transform=transform, regularization="weight_decay", 
            wd_lambda=10.0 ** k, x_test=data["test"][:, :-1], 
            y_test=data["test"][:, -1]
        )
        print(f"  {k=}: {E_in=:.3f}, {E_out=:.3f}")