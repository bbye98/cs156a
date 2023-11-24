import numpy as np
import pandas as pd
from sklearn import svm

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