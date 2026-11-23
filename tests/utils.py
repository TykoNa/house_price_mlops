import numpy as np


def generate_random_data(noise_lvl=0.1):
    num_points = 100
    X = 10 * np.random.random(size=num_points)
    return X


def generate_fixed_data():
    num_points = 100
    X = np.linspace(1, 10, num_points)
    y = X + 3
    return X, y


class NoFit:
    pass


class BadFit:
    fit = "not a function"
