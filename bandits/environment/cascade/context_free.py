from typing import Any, TypedDict, Union

import gymnasium as gym
import numpy as np
from sklearn.utils import check_random_state

from bandits.environment.cascade.shared_utils import (
    ActionRecommendation,
    get_optimal_ordering,
    get_prob_of_a_click,
)


class Reward(TypedDict):
    reward: float  # 1/0 for if anything was clicked
    position_of_click: Union[int, None]  # position of the click. None if reward is 0
    prob_of_click: float  # Want to maximise this prob


class CascadeContextFreeBandit(gym.Env):
    def __init__(self, weights: np.ndarray, max_steps: int = 10_000, len_list: int = 1):
        self.weights = weights
        self.n_actions = len(weights)
        self.len_list = len_list
        self.max_steps = max_steps
        self.observation_space = None
        self.action_space = ActionRecommendation(
            n_actions=self.n_actions, len_list=self.len_list
        )

        self.optimal_action = get_optimal_ordering(weights, self.len_list)
        self.optimal_reward = get_prob_of_a_click(
            weights=weights, action=self.optimal_action
        )
        self.optimal_weights = self.weights[self.optimal_action]

    def _get_obs(self) -> np.ndarray:
        return None

    def _get_info(self, reward: Reward = None) -> dict[str, Any]:
        if reward is None:
            return {}

        return dict(
            position_of_click=reward["position_of_click"],
            prob_of_click=reward["prob_of_click"],
            optimal_action=self.optimal_action,
            optimal_reward=self.optimal_reward,
            optimal_weights=self.optimal_weights,
        )

    def _get_click(self, action: list[int]) -> tuple[int, Union[int, None]]:
        reward = 0
        position_of_click = None

        for a in action:
            if self.random_.rand() < self.weights[a]:
                reward = 1
                position_of_click = a

                return reward, position_of_click
        return reward, position_of_click

    def _get_rewards(self, action: list[int]) -> Reward:
        reward, position_of_click = self._get_click(action=action)
        prob_of_click = get_prob_of_a_click(weights=self.weights, action=action)

        return dict(
            reward=reward,
            position_of_click=position_of_click,
            prob_of_click=prob_of_click,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.random_ = check_random_state(seed)
        self.action_space.seed(seed)
        self._n_steps = 0
        observation = self._get_obs()
        info = self._get_info(reward=None)
        return observation, info

    def step(self, action: int):
        reward = self._get_rewards(action=action)
        terminated = False
        truncated = False
        self._n_steps += 1

        if self._n_steps >= self.max_steps:
            truncated = True

        observation = self._get_obs()
        info = self._get_info(reward)

        return observation, reward["reward"], terminated, truncated, info
