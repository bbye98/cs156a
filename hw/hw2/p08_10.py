import numpy as np
import pandas as pd

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

def target_function_homework_2(x):
    f = lambda x: np.sign((x[:, -2:] ** 2).sum(axis=1) - 0.6)
    return f if x is None else f(x)

def generate_data(
        N, f, d=2, lb=-1.0, ub=1.0, *, bias=False, rng=None, seed=None):
    if rng is None:
        rng = np.random.default_rng(seed)
    x = rng.uniform(lb, ub, (N, d))
    if bias:
        x = np.hstack((np.ones((N, 1)), x))
    return x, f(x)

def validate_binary(w, x, y):
    return np.count_nonzero(np.sign(x @ w) != y, axis=0) / x.shape[0]

if __name__ == "__main__":
    rng = np.random.default_rng()

    N_train = N_runs = 1_000
    N_test = 9 * N_train
    noise = (0.1, lambda y: -y)
    reg = LinearRegression(vf=validate_binary, noise=noise, rng=rng)
    E_in = 0
    for _ in range(N_runs):
        x_train, y_train = generate_data(N_train, target_function_homework_2,
                                         bias=True, rng=rng)
        x_test, y_test = generate_data(N_test, target_function_homework_2,
                                       bias=True, rng=rng)
        E_in += reg.train(x_train, y_train)
    print("\n[Homework 2 Problem 8]\n"
          f"For the linear regression model with {noise[0]:.0%} noise, "
          f"the average in-sample error over {N_runs:,} runs is "
          f"{E_in / N_runs:.6f}.")

    transform = lambda x: np.hstack((x, x[:, 1:2] * x[:, 2:], x[:, 1:2] ** 2,
                                     x[:, 2:] ** 2))
    gs = np.array(((-1, -0.05, 0.08, 0.13, 1.5, 1.5), 
                   (-1, -0.05, 0.08, 0.13, 1.5, 15),
                   (-1, -0.05, 0.08, 0.13, 15, 1.5),
                   (-1, -1.5, 0.08, 0.13, 0.05, 0.05),
                   (-1, -0.05, 0.08, 1.5, 0.15, 0.15)))
    w = np.zeros_like(gs[0])
    reg.set_parameters(vf=validate_binary, transform=transform, noise=noise, 
                       update=True)
    for _ in range(N_runs):
        x_train, y_train = generate_data(N_train, target_function_homework_2,
                                         bias=True, rng=rng)
        reg.train(x_train, y_train)
        w += reg.w
    w /= N_runs
    counters = np.zeros(6, dtype=float)
    for _ in range(N_runs):
        x_test, y_test = generate_data(N_test, target_function_homework_2,
                                    bias=True, rng=rng)
        x_test = transform(x_test)
        y_test[rng.choice(N_test, round(noise[0] * N_test), False)] *= -1
        h_test = np.sign(x_test @ w)
        counters += (*validate_binary(gs.T, x_test, h_test[:, None]),
                    np.count_nonzero(h_test != y_test) / N_test)
    counters /= N_runs
    df = pd.DataFrame({
        "choice": [f"[{chr(97 + i)}]" for i in range(5)],
        "g": [f"[{', '.join(f'{c:.2g}' for c in g)}]" for g in gs],
        "probability": 1 - counters[:5]
    })
    print("\n[Homework 2 Problem 9]\n"
          f"The average weight vector over {N_runs:,} runs is "
          "w = [", ", ".join(f"{v:.6f}" for v in w), "].\n", 
          df.to_string(index=False), sep="")

    print("\n[Homework 2 Problem 10]\n"
          f"The average out-of-sample error over {N_runs:,} runs is "
          f"{counters[5]:.6f}.")