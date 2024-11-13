from typing import Any, TypedDict, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bandits.environment.cascade.shared_utils import (
    ActionRecommendation,
    get_optimal_ordering,
    get_prob_of_a_click,
)


class Reward(TypedDict):
    reward: float  # 1/0 for if anything was clicked
    position_of_click: Union[int, None]  # position of the click. None if reward is 0
    prob_of_click: float  # Want to maximise this prob


class CascadeContextualBandit(gym.Env):
    def __init__(self, weights: np.ndarray, max_steps: int = 10_000, len_list: int = 1):
        self.weights = weights
        self.dim = weights.shape[0]
        self.n_actions = len(weights)
        self.len_list = len_list
        self.max_steps = max_steps
        self.observation_space = spaces.Discrete(self.dim)
        self.contexts = np.identity(self.dim)
        self.context = None
        self.action_space = ActionRecommendation(
            n_actions=self.n_actions, len_list=self.len_list
        )

        self.optimal_action = {
            context: get_optimal_ordering(weights[context], self.len_list)
            for context in range(self.dim)
        }
        self.optimal_reward = {
            context: get_prob_of_a_click(
                self.weights[context], action=self.optimal_action[context]
            )
            for context in range(self.dim)
        }
        self.optimal_weights = {
            context: self.weights[context, self.optimal_action[context]]
            for context in range(self.dim)
        }

    def _get_obs(self) -> np.ndarray:
        self.context = self.observation_space.sample()
        return self.contexts[self.context]

    def _get_info(self, reward: Reward = None) -> dict[str, Any]:
        if reward is None:
            return {}

        return dict(
            position_of_click=reward["position_of_click"],
            prob_of_click=reward["prob_of_click"],
            optimal_action=self.optimal_action[self.context],
            optimal_reward=self.optimal_reward[self.context],
            optimal_weights=self.optimal_weights[self.context],
        )

    def _get_click(self, action: list[int]) -> tuple[int, Union[int, None]]:
        reward = 0
        position_of_click = None

        for a in action:
            if np.random.rand() < self.weights[self.context][a]:
                reward = 1
                position_of_click = a

                return reward, position_of_click
        return reward, position_of_click

    def _get_rewards(self, action: list[int]) -> Reward:
        reward, position_of_click = self._get_click(action=action)
        prob_of_click = get_prob_of_a_click(
            weights=self.weights[self.context], action=action
        )

        return dict(
            reward=reward,
            position_of_click=position_of_click,
            prob_of_click=prob_of_click,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)
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
