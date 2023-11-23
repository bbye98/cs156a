import numpy as np
import pandas as pd

def generate_data(
        N, f, d=2, lb=-1.0, ub=1.0, *, bias=False, rng=None, seed=None):
    if rng is None:
        rng = np.random.default_rng(seed)
    x = rng.uniform(lb, ub, (N, d))
    if bias:
        x = np.hstack((np.ones((N, 1)), x))
    return x, f(x)

if __name__ == "__main__":
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
        df.loc[len(df)] = (f"[{chr(97 + i)}]", h, fmt(a_avg, b_avg), bias, var)
    print(f"\n[Homework 4 Problems 4–7]\n{df.to_string(index=False)}")