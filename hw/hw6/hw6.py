#!/usr/bin/env python3

"""
Benjamin Ye
CS/CNE/EE 156a: Learning Systems (Fall 2023)
November 6, 2023
Homework 6
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests

CWD = Path(__file__).resolve()
sys.path.insert(0, str(CWD.parents[2]))
from cs156a import LinearRegression, validate_binary

DATA_DIR = (CWD.parents[2] / "data").resolve()

if __name__ == "__main__":
    rng = np.random.default_rng()

    ### Problems 2–6 ###########################################################

    DATA_DIR.mkdir(exist_ok=True)
    data = {"train": "in.dta", "test": "out.dta"}
    for dataset, file in data.items():
        if not (DATA_DIR / file).exists():
            r = requests.get(f"http://work.caltech.edu/data/{file}")
            with open(DATA_DIR / file, "wb") as f:
                f.write(r.content)
        data[dataset] = np.loadtxt(DATA_DIR / file)

    transform = lambda x: np.hstack((
        np.ones((len(x), 1), dtype=float), 
        x, 
        x[:, :1] ** 2, 
        x[:, 1:] ** 2, 
        np.prod(x, axis=1, keepdims=True),
        np.abs(x[:, :1] - x[:, 1:]), 
        np.abs(x[:, :1] + x[:, 1:])
    ))
    reg = LinearRegression(vf=validate_binary, transform=transform, rng=rng)
    E_in = reg.train(data["train"][:, :-1], data["train"][:, -1])
    E_out = reg.get_error(data["test"][:, :-1], data["test"][:, -1])
    print("\n[Homework 6 Problem 2]\n",
          "For the linear regression model without regularization, the "
          f"in-sample and out-of-sample errors are {E_in:.5f} and "
          f"{E_out:.5f}, respectively.", sep="")
    
    df = pd.DataFrame(columns=["k", "in-sample error", "out-of-sample error"])
    for k in np.arange(-5, 7):
        reg.set_parameters(regularization="weight_decay",
                           weight_decay_lambda=10.0 ** k, update=True)
        E_in = reg.train(data["train"][:, :-1], data["train"][:, -1])
        df.loc[len(df)] = k, E_in, reg.get_error(data["test"][:, :-1],
                                                 data["test"][:, -1])
    print("\n[Homework 6 Problems 3–6]\n"
          "Linear regression with weight decay regularization:\n",
          df.to_string(index=False), sep="")