"""
Sparse Reward Grid World Environment
A simple grid world where the agent must find a goal location.
Only receives reward upon reaching the goal (sparse reward).
"""

import numpy as np
from typing import Tuple, Optional


class GridWorldEnv:
    """
    Grid world environment with sparse rewards.
    Agent starts at (0, 0) and must reach the goal.
    Only receives +1 reward when reaching the goal, 0 otherwise.
    """

    def __init__(self, size: int = 10, goal_pos: Optional[Tuple[int, int]] = None):
        self.size = size
        self.goal_pos = goal_pos if goal_pos else (size - 1, size - 1)
        self.state = None
        self.action_space = 4  # up, down, left, right
        self.observation_space = size * size  # flattened grid position

    def reset(self) -> np.ndarray:
        """Reset environment to starting position."""
        self.state = np.array([0, 0])
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """Convert state to one-hot encoded observation."""
        obs = np.zeros(self.observation_space)
        idx = self.state[0] * self.size + self.state[1]
        obs[idx] = 1.0
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take action in environment.

        Actions:
        0: up, 1: down, 2: left, 3: right
        """
        # Store old position
        old_state = self.state.copy()

        # Apply action
        if action == 0:  # up
            self.state[0] = max(0, self.state[0] - 1)
        elif action == 1:  # down
            self.state[0] = min(self.size - 1, self.state[0] + 1)
        elif action == 2:  # left
            self.state[1] = max(0, self.state[1] - 1)
        elif action == 3:  # right
            self.state[1] = min(self.size - 1, self.state[1] + 1)

        # Check if goal reached (sparse reward)
        done = (self.state[0] == self.goal_pos[0] and
                self.state[1] == self.goal_pos[1])
        reward = 1.0 if done else 0.0

        return self._get_observation(), reward, done, {}

    def render(self):
        """Print current state."""
        grid = np.zeros((self.size, self.size), dtype=str)
        grid[:] = '.'
        grid[self.goal_pos[0], self.goal_pos[1]] = 'G'
        grid[self.state[0], self.state[1]] = 'A'

        for row in grid:
            print(' '.join(row))
        print()
