from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).parents[2] / "data"

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