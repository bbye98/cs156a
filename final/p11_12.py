from cvxopt import matrix, solvers
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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

if __name__ == "__main__":
    x = np.array(((1, 0), (0, 1), (0, -1), (-1, 0), (0, 2), (0, -2), (-2, 0)),
                 dtype=float)
    y = np.array((-1, -1, -1, 1, 1, 1, 1), dtype=float)
    z = np.hstack((x[:, 1:] ** 2 - 2 * x[:, :1] - 1, 
                   x[:, :1] ** 2 - 2 * x[:, 1:] + 1))

    ticks = np.arange(-6, 7)
    _, ax = plt.subplots()
    ax.plot(*z[y == 1].T, "s", label="$+1$")
    ax.plot(*z[y == -1].T, "o", label="$-1$")
    ax.grid(ls=":")
    ax.legend(title="classification", loc="lower left")
    ax.set_aspect("equal", "box")
    ax.set_xlabel("$z_1$")
    ax.set_xlim(-6, 6)
    ax.set_xticks(ticks)
    ax.set_ylabel("$z_2$")
    ax.set_ylim(-6, 6)
    ax.set_yticks(ticks)
    plt.show()

    solvers.options["show_progress"] = False
    solution = solvers.qp(matrix(np.outer(y, y) * (1 + x @ x.T) ** 2),
                          matrix(-np.ones((x.shape[0], 1))), 
                          matrix(-np.eye(x.shape[0])), 
                          matrix(np.zeros(x.shape[0])), 
                          matrix(y[None]), 
                          matrix(0.0))

    clf = svm.SVC(C=np.finfo(float).max, kernel="poly", degree=2, gamma=1, 
                  coef0=1)
    clf.fit(x, y)

    print("\n[Final Exam Problem 11]\n"
          "The second-order polynomial hard margin support vector "
          "machine (SVM) uses\n  - cvxopt.solvers.qp: "
          f"{(~np.isclose(solution['x'], 0)).sum()} support vectors;\n"
          f"  - sklearn.svm.SVC: {clf.n_support_.sum()} support vectors.")