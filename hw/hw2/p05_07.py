import numpy as np

class Perceptron:
    def __init__(self, w=None, *, vf=None):
        self.set_parameters(w, vf=vf)

    def get_error(self, x, y):
        if self.vf is not None and self.w is not None:
            return self.vf(self.w, x, y)

    def set_parameters(self, w=None, *, vf=None, update=False) -> None:
        if update:
            self.vf = vf or self.vf
            self._w = self._w if w is None else w
        else:
            self.vf = vf
            self._w = w

    def train(self, x, y):
        self.iters = 0
        self.w = (np.zeros(x.shape[1], dtype=float) if self._w is None 
                  else self._w)
        while True:
            wrong = np.argwhere(np.sign(x @ self.w) != y)[:, 0]
            if wrong.size == 0:
                break
            index = np.random.choice(wrong)
            self.w += y[index] * x[index]
            self.iters += 1
        if self.vf:
            return self.vf(self.w, x, y)

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

def target_function_random_line(x=None, *, rng=None, seed=None):
    if rng is None:
        rng = np.random.default_rng(seed)
    line = rng.uniform(-1, 1, (2, 2))
    f = lambda x: np.sign(
        x[:, -1] - line[0, 1] 
        - np.divide(*(line[1] - line[0])[::-1]) * (x[:, -2] - line[0, 0])
    )
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

    N_train = 100
    N_test = 9 * N_train
    N_runs = 1_000
    f = target_function_random_line(rng=rng)
    reg = LinearRegression(vf=validate_binary, rng=rng)
    errors = np.zeros(2, dtype=float)
    for _ in range(N_runs):
        E_in = reg.train(*generate_data(N_train, f, bias=True, rng=rng))
        errors += (
            E_in, 
            reg.get_error(*generate_data(N_test, f, bias=True, rng=rng))
        )
    errors /= N_runs
    print("\n[Homework 2 Problems 5–6]\n"
          "For the linear regression model, the average in-sample and "
          f"out-of-sample errors over {N_runs:,} runs are "
          f"{errors[0]:.6f} and {errors[1]:.6f}, respectively.")

    N_train = 10
    pla = Perceptron(vf=validate_binary)
    iters = 0
    for _ in range(N_runs):
        f = target_function_random_line(rng=rng)
        x_train, y_train = generate_data(N_train, f, bias=True, rng=rng)
        reg.train(x_train, y_train)
        pla.set_parameters(w=reg.w, update=True)
        pla.train(x_train, y_train)
        iters += pla.iters
    print("\n[Homework 2 Problem 7]\n"
          "With initial weights from linear regression, the perceptron "
          f"takes an average of {iters / N_runs:.0f} iterations to "
          "converge.")