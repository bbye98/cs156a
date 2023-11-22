import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import requests
from sklearn import svm

mpl.rcParams.update(
    {
        "axes.labelsize": 14,
        "figure.autolayout": True,
        "figure.figsize": (4.875, 3.65625),
        "font.size": 12,
        "legend.columnspacing": 1,
        "legend.edgecolor": "1",
        "legend.framealpha": 0,
        "legend.fontsize": 12,
        "legend.handlelength": 1.25,
        "legend.labelspacing": 0.25,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "text.usetex": True
    }
)

CWD = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(CWD.parents[1]))
from cs156a import validate_binary, linear_regression

DATA_DIR = (CWD.parents[1] / "data").resolve()

if __name__ == "__main__":
    rng = np.random.default_rng()

    # Regularized Linear Regression
    DATA_DIR.mkdir(exist_ok=True)
    data = {}
    for dataset in ["train", "test"]:
        file = f"features.{dataset}"
        if not (DATA_DIR / file).exists():
            r = requests.get(f"http://www.amlbook.com/data/zip/{file}")
            with open(DATA_DIR / file, "wb") as f:
                f.write(r.content)
        data[dataset] = np.loadtxt(DATA_DIR / file)

    # Problems 7–10
    print("\n[FE P7–9]\nLinear regression with regularization:")
    transform = lambda x: np.hstack((x, x[:, 1:2] * x[:, 2:], x[:, 1:2] ** 2,
                                     x[:, 2:] ** 2))
    for digit in range(10):
        x = np.hstack((np.ones((len(data["train"]), 1), dtype=float), 
                    data["train"][:, 1:]))
        y = 2 * (data["train"][:, 0] == digit) - 1
        x_test = np.hstack((np.ones((len(data["test"]), 1), dtype=float), 
                            data["test"][:, 1:]))
        y_test = 2 * (data["test"][:, 0] == digit) - 1
        print(f"  {digit} vs. all:")
        for t, l in zip((None, transform), ("X", "Z")):
            E_in, E_out = linear_regression(
                vf=validate_binary, x=x, y=y, transform=t, 
                regularization="weight_decay", wd_lambda=1,
                x_test=x_test, y_test=y_test, rng=rng
            )
            print(f"    {l}: {E_in=:.6f}, {E_out=:.6f}")

    print("\n[FE P10]\nLinear regression with transform and "
          "regularization for 1 vs. 5 classifier:")
    subset = data["train"][np.isin(data["train"][:, 0], (1, 5))]
    x = np.hstack((np.ones((len(subset), 1), dtype=float), subset[:, 1:]))
    y = (subset[:, 0] == 1).astype(int) - (subset[:, 0] == 5)
    subset_test = data["test"][np.isin(data["test"][:, 0], (1, 5))]
    x_test = np.hstack((np.ones((len(subset_test), 1), dtype=float),
                        subset_test[:, 1:]))
    y_test = (subset_test[:, 0] == 1).astype(int) - (subset_test[:, 0] == 5)
    for wd_lambda in (0.01, 1):
        E_in, E_out = linear_regression(
            vf=validate_binary, x=x, y=y, transform=transform, 
            regularization="weight_decay", wd_lambda=wd_lambda, 
            x_test=x_test, y_test=y_test, rng=rng
        )
        print(f"    lambda={wd_lambda}: {E_in=:.6f}, {E_out=:.6f}")

    # Problems 11–12
    x = np.array(((1, 0), (0, 1), (0, -1), (-1, 0), (0, 2), (0, -2), 
              (-2, 0)), dtype=float)
    y = np.array((-1, -1, -1, 1, 1, 1, 1), dtype=int)
    z = np.hstack((x[:, 1:] ** 2 - 2 * x[:, :1] - 1, 
                   x[:, :1] ** 2 - 2 * x[:, 1:] + 1))

    _, ax = plt.subplots()
    ax.scatter(*z[y == 1].T, marker="s", label="$+1$")
    ax.scatter(*z[y == -1].T, marker="o", label="$-1$")
    ax.set_aspect("equal", "box")
    ax.set_xlabel("$z_1$")
    ax.set_xlim(-6, 6)
    ax.set_ylabel("$z_2$")
    ax.set_ylim(-6, 6)
    ax.legend(title="Classification", loc="lower left")
    plt.show()

    clf = svm.SVC(C=np.finfo(float).max, kernel="poly", degree=2, gamma=1, 
                  coef0=1)
    clf.fit(x, y)
    print("[FE P12]\nSecond-order polynomial support vector machine:\n"
        f"  N_sv={clf.n_support_.sum()}")