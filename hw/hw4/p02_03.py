import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

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

def vapnik_chervonenkis_bound(m_H, N, delta):
    return np.sqrt(8 * np.log(4 * m_H(2 * N) / delta) / N)

def rademacher_bound(m_H, N, delta):
    return (np.sqrt(2 * np.log(2 * N * m_H(N)) / N) 
            + np.sqrt(2 * np.log(1 / delta) / N) + 1 / N)

def parrondo_van_den_broek_bound(m_H, N, delta, ub=10.0):
    return np.vectorize(
        lambda N: optimize.root_scalar(
            lambda eps: np.sqrt((2 * eps + np.log(6 * m_H(2 * N) / delta)) / N)
                        - eps, 
            bracket=(0.0, ub), method="toms748"
        ).root
    )(N)

def devroye_bound(m_H, N, delta, ub=10.0):
    return np.vectorize(
        lambda N: optimize.root_scalar(
            lambda eps, N: np.sqrt(
                (4 * eps * (1 + eps) + np.log(4 / delta) + np.log(m_H(N ** 2)))
                / (2 * N)
            ) - eps,
            args=N, 
            bracket=(0.0, ub), 
            method="toms748"
        ).root
    )(N)

if __name__ == "__main__":
    d_vc = 50
    delta = 0.05
    m_H = lambda N: N ** d_vc
    Ns = np.arange(3, 10_001, dtype=np.longdouble)
    bounds = {
        "Vapnik–Chervonenkis": vapnik_chervonenkis_bound(m_H, Ns, delta),
        "Rademacher": rademacher_bound(m_H, Ns, delta),
        "Parrondo–van den Broek": parrondo_van_den_broek_bound(m_H, Ns, delta),
        "Devroye": devroye_bound(m_H, Ns, delta)
    }
    df = pd.DataFrame(columns=["N"] + [b for b in bounds.keys()])
    _, ax = plt.subplots()
    for label, bound in bounds.items():
        ax.plot(Ns, bound, label=label)
    ax.set_yscale("log")
    ylim = ax.get_ylim()
    for N in (10_000, 5):
        df.loc[len(df)] = N, *[bound[np.where(Ns == N)[0][0]] 
                               for bound in bounds.values()]
        ax.plot((N, N), ylim, "k:")
    ax.legend(title="Generalization bound")
    ax.set_xlabel("$N$")
    ax.set_xscale("log")
    ax.set_ylabel("$\epsilon$")
    ax.set_ylim(ylim)
    plt.show()
    print("\n[Homework 4 Problems 2–3]\nGeneralization bounds:\n",
          df.to_string(index=False), sep="")