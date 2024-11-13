from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from sklearn.utils import check_random_state


@dataclass
class BaseContextFreeTS(metaclass=ABCMeta):
    n_actions: int
    len_list: int = 1
    batch_size: int = 1
    random_state: Optional[int] = None

    def __post_init__(self) -> None:
        self.random_ = check_random_state(self.random_state)
        self.n_trial = 0
        self.action_counts = np.zeros(self.n_actions, dtype=np.int64)
        self.reward_counts = np.zeros(self.n_actions, dtype=np.int64)
        self.action_counts_temp = np.zeros(self.n_actions, dtype=np.int64)
        self.reward_counts_temp = np.zeros(self.n_actions, dtype=np.int64)

    @abstractmethod
    def select_action(self) -> np.ndarray:
        """Select a list of actions."""
        raise NotImplementedError

    def update_params(self, action: int, reward: float) -> None:
        """Update policy parameters.

        Parameters
        ----------
        action: int
            Selected action by the policy.

        reward: float
            Observed reward for the chosen action and position.
        """
        self.n_trial += 1
        self.action_counts_temp[action] += 1
        self.reward_counts_temp[action] += reward
        if self.n_trial % self.batch_size == 0:
            self.action_counts = np.copy(self.action_counts_temp)
            self.reward_counts = np.copy(self.reward_counts_temp)

    def cascade_params_update(
        self, action: list[int], reward_position: Union[int, None] = None
    ) -> None:
        """Update policy parameters assuming a cascade reward structure.

        Parameters
        ----------
        action: list[int]
            Selected action by the policy.

        reward_position: Union[int, None]
            The position where the reward was observed,
            which usually is a click or some other
            binary reward measure.
        """

        for action_case in action:
            if reward_position == action_case:
                self.update_params(action=action_case, reward=1)
                break
            else:
                self.update_params(action=action_case, reward=0)


@dataclass
class BernoulliTS(BaseContextFreeTS):
    alpha: Optional[np.ndarray] = None
    beta: Optional[np.ndarray] = None

    def __post_init__(self):
        self.alpha = np.ones(self.n_actions) if self.alpha is None else self.alpha
        self.beta = np.ones(self.n_actions) if self.beta is None else self.beta

        super().__post_init__()

    def select_action(self) -> np.ndarray:
        """Select a list of actions.

        Returns
        ----------
        selected_actions: array-like, shape (len_list, )
            List of selected actions.

        """
        predicted_rewards = self.random_.beta(
            a=self.reward_counts + self.alpha,
            b=(self.action_counts - self.reward_counts) + self.beta,
        )
        return predicted_rewards.argsort()[::-1][: self.len_list]


@dataclass
class Random(BaseContextFreeTS):
    def __post_init__(self):
        super().__post_init__()

    def select_action(self) -> np.ndarray:
        """Select a list of actions.

        Returns
        ----------
        selected_actions: array-like, shape (len_list, )
            List of selected actions.

        """
        return self.random_.choice(
            range(self.n_actions), size=self.len_list, replace=False
        )
