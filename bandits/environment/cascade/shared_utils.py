import gymnasium as gym
import numpy as np


# Define your custom action space class
# NOTE: https://gym-docs.readthedocs.io/en/latest/content/spaces.html#general-functions
# FOR CREATING CUSTOM GYM ACTION SPACES NEEED TO DEFINE MANY METHODS!
# BUT NOT GOING TO DO THIS JUST FOR THE KEY ONES!
class ActionRecommendation(gym.spaces.Space):
    def __init__(self, n_actions: int, len_list: int):
        self.n_actions = n_actions
        self.len_list = len_list

    def sample(self) -> list[int]:
        return np.random.choice(
            range(self.n_actions), replace=False, size=self.len_list
        )

    @property
    def n(self):
        return self.n_actions


def get_optimal_ordering(weights: np.ndarray, len_list: int) -> list[int]:
    return weights.argsort()[::-1][:len_list]


def get_prob_of_a_click(weights: np.ndarray, action: list[int]) -> float:
    non_click_prob = 1
    for a in action:
        non_click_prob *= 1 - weights[a]

    prob_of_click = 1 - non_click_prob
    return prob_of_click
