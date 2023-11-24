import numpy as np

if __name__ == "__main__":
    rng = np.random.default_rng()

    x = rng.uniform(size=(10_000_000, 2))
    e_1, e_2 = x.mean(axis=0)
    e = x.min(axis=1).mean()
    print("\n[Homework 7 Problem 6]\n"
          "The expected values for paired independent uniform random "
          f"variables and their minimum are {e_1:.6f}, "
          f"{e_2:.6f}, and {e:.6f}, respectively.")