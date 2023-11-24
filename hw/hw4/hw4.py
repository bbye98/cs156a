#!/usr/bin/env python3

"""
Benjamin Ye
CS/CNE/EE 156a: Learning Systems (Fall 2023)
October 23, 2023
Homework 4
"""

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
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

    ### Problems 2–3 ##########################################################

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

    ### Problems 4–7 ##########################################################

    n_runs = 10_000_000
    f = lambda x: np.sin(np.pi * x)
    hs = {
        # h(x): ((a(x_in, y_in), b(x_in, y_in)),
        #        bias(x_out, y_out, a_avg, b_avg),
        #        var(x_out, a, a_avg, b, b_avg), 
        #        fmt(a_avg, b_avg))
        "b": (
            lambda x, y: (None, (y[::2] + y[1::2]) / 2),
            lambda xt, yt, ah, bh: ((bh - yt) ** 2).mean(),
            lambda xt, a, ah, b, bh: ((np.tile(b, (2, 1)) - bh) ** 2).mean(),
            lambda ah, bh: f"{bh:.2f}"
        ), 
        "ax": (
            lambda x, y: (
                (x[::2] * y[::2] + x[1::2] * y[1::2]) 
                / (x[::2] ** 2 + x[1::2] ** 2),
                None
            ),
            lambda xt, yt, ah, bh: ((ah * xt - yt) ** 2).mean(),
            lambda xt, a, ah, b, bh: (
                ((np.tile(a, (2, 1)) - ah) * xt) ** 2
            ).mean(),
            lambda ah, bh: f"{ah:.2f}x"
        ),
        "ax+b": (
            lambda x, y: (
                (y[::2] - y[1::2]) / (x[::2] - x[1::2]),
                (x[::2] * y[1::2] - x[1::2] * y[::2]) / (x[::2] - x[1::2])
            ),
            lambda xt, yt, ah, bh: ((ah * xt + bh - yt) ** 2).mean(),
            lambda xt, a, ah, b, bh: (
                ((np.tile(a, (2, 1)) - ah) * xt + np.tile(b, (2, 1)) - bh) ** 2
            ).mean(),
            lambda ah, bh: f"{ah:.2f}x{'+' if bh >= 0 else ''}{bh:.2f}"
        ),
        "ax^2": (
            lambda x, y: (
                (x[::2] * y[::2] + x[1::2] * y[1::2]) 
                / (x[::2] ** 3 + x[1::2] ** 3),
                None
            ),
            lambda xt, yt, ah, bh: ((ah * xt ** 2 - yt) ** 2).mean(),
            lambda xt, a, ah, b, bh: (
                ((np.tile(a, (2, 1)) - ah) * xt ** 2) ** 2
            ).mean(),
            lambda ah, bh: f"{ah:.2f}x^2"
        ),
        "ax^2+b": (
            lambda x, y: (
                (y[::2] - y[1::2]) / (x[::2] ** 2 - x[1::2] ** 2),
                (x[::2] ** 2 * y[1::2] - x[1::2] ** 2 * y[::2]) 
                / (x[::2] ** 2 - x[1::2] ** 2)
            ),
            lambda xt, yt, ah, bh: ((ah * xt ** 2 + bh - yt) ** 2).mean(),
            lambda xt, a, ah, b, bh: (
                ((np.tile(a, (2, 1)) - ah) * xt ** 2 
                 + np.tile(b, (2, 1)) - bh) ** 2
            ).mean(),
            lambda ah, bh: f"{ah:.2f}x^2{'+' if bh >= 0 else ''}{bh:.2f}"
        )
    }
    x_train, y_train = generate_data(2 * n_runs, f, 1)
    x_test, y_test = generate_data(2 * n_runs, f, 1)
    df = pd.DataFrame(columns=["choice", "h(x)", "g(x)", "bias", "variance"])
    for i, (h, (f_ab, f_bias, f_var, fmt)) in enumerate(hs.items()):
        as_, bs = f_ab(x_train, y_train)
        a_avg = None if as_ is None else as_.mean()
        b_avg = None if bs is None else bs.mean()
        bias = f_bias(x_test, y_test, a_avg, b_avg)
        var = f_var(x_test, as_, a_avg, bs, b_avg)
        df.loc[len(df)] = f"[{chr(97 + i)}]", h, fmt(a_avg, b_avg), bias, var
    print("\n[Homework 4 Problems 4–7]\n"
          "Learning algorithms for f(x)=sin(pi*x):\n",
          df.to_string(index=False), sep="")