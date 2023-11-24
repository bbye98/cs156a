import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.cluster import k_means

RNG = np.random.default_rng()

class RBFRegular:
    def __init__(self, gamma, K, *, vf) -> None:
        self.set_parameters(gamma, K, vf=vf)

    def get_error(self, x, y):
        if self.vf is not None and self.w is not None:
            return self.vf(self.w, self.get_phi(x, self.centers), y)

    def get_phi(self, x, centers):
        return np.hstack((
            np.ones((x.shape[0], 1), dtype=float), 
            np.exp(-self.gamma 
                   * np.linalg.norm((x[:, None] - centers), axis=2) ** 2)
        ))

    def set_parameters(self, gamma, K, *, vf=None, update=False) :
        if update:
            self.gamma = gamma or self.gamma
            self.K = K or self.K
            self.vf = vf or self.vf
        else:
            self.gamma = gamma
            self.K = K
            self.vf = vf

    def train(self, x: np.ndarray[float], y: np.ndarray[float]) -> float:
        self.centers = k_means(x, self.K, n_init="auto")[0]
        phi = self.get_phi(x, self.centers)
        self.w = np.linalg.pinv(phi) @ y
        if self.vf is not None:
            return self.vf(self.w, phi, y)

def target_function_final_exam(x):
    f = lambda x: np.sign(np.diff(x[:, -2:], axis=1)[:, 0]
                          + 0.25 * np.sin(np.pi * x[:, -1]))
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
    N_runs = 100
    N_train = 100
    N_test = 1_000
    columns = ["gamma", "K", "nonseparable", "outperform", 
               "E_in=0", "E_in", "E_out"]
    df = pd.DataFrame(columns=columns)
    for gamma, K in [(1.5, 9), (1.5, 12), (2, 9)]:
        clf = svm.SVC(C=np.finfo(float).max, gamma=gamma)
        rbf = RBFRegular(gamma, K, vf=validate_binary)
        counters = np.zeros(5, dtype=float) 
        for _ in range(N_runs):
            x_train, y_train = generate_data(N_train, 
                                             target_function_final_exam,
                                             rng=RNG)
            x_test, y_test = generate_data(N_test, target_function_final_exam, 
                                           rng=RNG)
            clf.fit(x_train, y_train)
            if not np.isclose(clf.score(x_train, y_train), 1):
                counters[0] += 1
                continue
            E_in = rbf.train(x_train, y_train)
            E_out = rbf.get_error(x_test, y_test)
            counters[1:5] += (1 - clf.score(x_test, y_test) < E_out, 
                              E_in == 0, E_in, E_out)
        counters /= N_runs
        df.loc[len(df)] = gamma, K, *counters
    print("\n[Final Exam Problems 13–18]\n"
          f"Radial basis function (RBF) model ({N_runs:,} runs):\n", 
          df.to_string(index=False), sep="")