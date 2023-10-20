#!/usr/bin/env python3

"""
Benjamin Ye
CS/CNE/EE 156a: Learning Systems (Fall 2023)
October 23, 2023
Homework 4
"""

import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, pathlib.Path(__file__).resolve().parent)
from cs156a import (
    vapnik_chervonenkis_bound, rademacher_bound, 
    parrondo_van_den_broek_bound, devroye_bound, generate_data
)

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
    # problems 2–3
    d_vc, delta = 50, 0.05
    m_H = lambda N: N ** d_vc
    Ns = np.arange(3, 10001, dtype=float)
    bounds = {
        "Vapnik–Chervonenkis": vapnik_chervonenkis_bound(m_H, Ns, delta),
        "Rademacher": rademacher_bound(m_H, Ns, delta),
        "Parrondo–van den Broek": parrondo_van_den_broek_bound(m_H, Ns, delta),
        "Devroye": devroye_bound(lambda N: d_vc * np.log(N), Ns, delta, log=True)
    }

    print(f"\n[HW4 P2–3]\nGeneralization bounds for {d_vc=} and {delta=}:")
    for N in (10000, 5):
        i = np.where(Ns == N)[0][0]
        print(f"  {N=:,}:")
        for l, b in bounds.items():
            print(f"    {l}: {b[i]:.3f}")

    _, ax = plt.subplots()
    for l, b in bounds.items():
        ax.plot(Ns, b, label=l)
    ax.set_yscale("log")
    ylim = ax.get_ylim()
    ax.plot((5, 5), ylim, "k:")
    ax.plot((10000, 10000), ylim, "k:")
    ax.legend(title="Generalization bound")
    ax.set_xlabel("$N$")
    ax.set_xscale("log")
    ax.set_ylabel("$\epsilon$")
    ax.set_ylim(ylim)
    ax.text(-0.2, 0.959, " ", transform=ax.transAxes)
    plt.show()

    # problems 4–6
    n_runs = 10_000_000
    x, y = generate_data(2 * n_runs, lambda x: np.sin(np.pi * x), 1)
    as_ = (x[::2] * y[::2] + x[1::2] * y[1::2]) / (x[::2] ** 2 + x[1::2] ** 2)
    a_avg = as_.mean()
    x, y = generate_data(2 * n_runs, lambda x: np.sin(np.pi * x), 1)
    print(f"\n[HW4 P4–6]\nBias and variance of h(x)=ax for f(x)=sin(pi*x):\n"
          f"  g(x)={a_avg:.2f}x, bias={((a_avg * x - y) ** 2).mean():.3f}, "
          f"var={(((np.tile(as_, (2, 1)) - a_avg) * x) ** 2).mean():.3f}")