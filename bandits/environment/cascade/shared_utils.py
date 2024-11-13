import numpy as np


def get_optimal_ordering(weights: np.ndarray, len_list: int) -> list[int]:
    return weights.argsort()[::-1][:len_list]


def get_prob_of_a_click(weights: np.ndarray, action: list[int]) -> float:
    non_click_prob = 1
    for a in action:
        non_click_prob *= 1 - weights[a]

    prob_of_click = 1 - non_click_prob
    return prob_of_click
