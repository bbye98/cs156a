from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = (Path(__file__).resolve().parents[2] / "data").resolve()

class LinearRegression:
    def __init__(
            self, *, vf=None, regularization=None, transform=None, noise=None,
            rng=None, seed=None, **kwargs):
        self.rng = np.random.default_rng(seed) if rng is None else rng
        self.set_parameters(vf=vf, regularization=regularization, 
                            transform=transform, noise=noise, **kwargs)

    def get_error(self, x, y):
        if self.transform:
            x = self.transform(x)
        if self.noise:
            N = x.shape[0]
            index = self.rng.choice(N, round(self.noise[0] * N), False)
            y[index] = self.noise[1](y[index])
        if self.vf is not None and self.w is not None:
            return self.vf(self.w, x, y)

    def set_parameters(
            self, *, vf=None, regularization=None, transform=None, noise=None,
            update=False, **kwargs):
        self._reg_params = {}       
        self.w = None
        if update:
            self.noise = noise or self.noise
            self.regularization = regularization or self.regularization
            if self.regularization == "weight_decay" \
                    and "weight_decay_lambda" in kwargs:
                self._reg_params["lambda"] = kwargs["weight_decay_lambda"]
            self.transform = transform or self.transform
            self.vf = vf or self.vf
        else:
            self.noise = noise
            self.regularization = regularization
            if regularization == "weight_decay":
                self._reg_params["lambda"] = kwargs["weight_decay_lambda"]
            self.transform = transform
            self.vf = vf

    def train(self, x, y):
        if self.transform:
            x = self.transform(x)
        if self.noise:
            N = x.shape[0]
            index = self.rng.choice(N, round(self.noise[0] * N), False)
            y[index] = self.noise[1](y[index])
        if self.regularization is None:
            self.w = np.linalg.pinv(x) @ y
        elif self.regularization == "weight_decay":
            self.w = np.linalg.inv(
                x.T @ x 
                + self._reg_params["lambda"] * np.eye(x.shape[1], dtype=float)
            ) @ x.T @ y
        if self.vf is not None:
            return self.vf(self.w, x, y)

def validate_binary(w, x, y):
    return np.count_nonzero(np.sign(x @ w) != y, axis=0) / x.shape[0]

if __name__ == "__main__":
    rng = np.random.default_rng()

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
        df.loc[len(df)] = (k, E_in, reg.get_error(data["test"][:, :-1],
                                                  data["test"][:, -1]))
    print(f"\n[Homework 6 Problems 3–6]\n{df.to_string(index=False)}")