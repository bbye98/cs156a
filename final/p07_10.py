from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).parents[1] / "data"
RNG = np.random.default_rng()

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
    DATA_DIR.mkdir(exist_ok=True)
    data = {}
    for dataset in ["train", "test"]:
        file = f"features.{dataset}"
        if not (DATA_DIR / file).exists():
            r = requests.get(f"http://www.amlbook.com/data/zip/{file}")
            with open(DATA_DIR / file, "wb") as f:
                f.write(r.content)
        data[dataset] = np.loadtxt(DATA_DIR / file)

    weight_decay_lambda = 1
    transform = lambda x: np.hstack((x, x[:, 1:2] * x[:, 2:], x[:, 1:2] ** 2, 
                                    x[:, 2:] ** 2))
    reg = LinearRegression(vf=validate_binary, regularization="weight_decay", 
                           weight_decay_lambda=weight_decay_lambda, rng=RNG)
    reg_transform = LinearRegression(vf=validate_binary,
                                     regularization="weight_decay", 
                                     transform=transform,
                                     weight_decay_lambda=weight_decay_lambda,
                                     rng=RNG)
    df = pd.DataFrame(columns=["classifier", "E_in", "E_out",
                               "transform E_in", "transform E_out"])
    for digit in range(10):
        x_train = np.hstack((np.ones((len(data["train"]), 1), dtype=float), 
                            data["train"][:, 1:]))
        y_train = 2 * (data["train"][:, 0] == digit) - 1
        x_test = np.hstack((np.ones((len(data["test"]), 1), dtype=float), 
                            data["test"][:, 1:]))
        y_test = 2 * (data["test"][:, 0] == digit) - 1
        E_in = reg.train(x_train, y_train)
        E_in_transform = reg_transform.train(x_train, y_train)
        df.loc[digit] = (f"{digit} vs. all", E_in,
                         reg.get_error(x_test, y_test), E_in_transform,
                         reg_transform.get_error(x_test, y_test))
    print("\n[Final Exam Problems 7–9]\n"
          "Linear regression with regularization "
          f"(lambda={weight_decay_lambda}):\n", df.to_string(index=False), 
          sep="")
    
    subset_train = data["train"][np.isin(data["train"][:, 0], (1, 5))]
    x_train = np.hstack((np.ones((subset_train.shape[0], 1), dtype=float), 
                         subset_train[:, 1:]))
    y_train = (subset_train[:, 0] == 1).astype(int) - (subset_train[:, 0] == 5)
    subset_test = data["test"][np.isin(data["test"][:, 0], (1, 5))]
    x_test = np.hstack((np.ones((subset_test.shape[0], 1), dtype=float),
                        subset_test[:, 1:]))
    y_test = (subset_test[:, 0] == 1).astype(int) - (subset_test[:, 0] == 5)
    df = pd.DataFrame(columns=["lambda", "in-sample error", 
                               "out-of-sample error"])
    for weight_decay_lambda in (0.01, 1):
        reg_transform.set_parameters(weight_decay_lambda=weight_decay_lambda, 
                                     update=True)
        E_in = reg_transform.train(x_train, y_train)
        df.loc[len(df)] = (weight_decay_lambda, E_in, 
                           reg_transform.get_error(x_test, y_test))
    print("\n[Final Exam Problem 10]\n"
          "Linear regression with regularization and transform for "
          "1 vs. 5 classifier:\n", df.to_string(index=False), sep="")