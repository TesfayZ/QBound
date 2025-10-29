"""
Adaptive QBound: Dynamic Q-value bounds that adjust during training

This module provides strategies for adapting Q-value bounds over time to balance
exploration in early training with refined bounds in later training.

Strategies:
1. Linear adaptation: Linearly interpolate from loose initial bounds to tight final bounds
2. Delayed activation: Start without QBound, enable after N episodes
3. Schedule-based: Custom scheduling functions for bound adaptation
"""

import numpy as np


class AdaptiveQBound:
    """
    Adaptive Q-value bounds that change during training.

    Bounds linearly interpolate from (Q_min_init, Q_max_init) to (Q_min_final, Q_max_final)
    over the course of training.

    Example:
        # Start with loose bounds, tighten over time
        adaptive = AdaptiveQBound(
            Q_min_init=-600, Q_max_init=200,
            Q_min_final=-200, Q_max_final=0
        )

        # At episode 100 of 2000
        Q_min, Q_max = adaptive.get_bounds(episode=100, total_episodes=2000)
    """

    def __init__(self, Q_min_init, Q_max_init, Q_min_final, Q_max_final):
        """
        Initialize adaptive bounds.

        Args:
            Q_min_init: Initial lower bound (typically more negative/permissive)
            Q_max_init: Initial upper bound (typically higher/permissive)
            Q_min_final: Final lower bound (tighter constraint)
            Q_max_final: Final upper bound (tighter constraint)
        """
        self.Q_min_init = Q_min_init
        self.Q_max_init = Q_max_init
        self.Q_min_final = Q_min_final
        self.Q_max_final = Q_max_final

    def get_bounds(self, episode, total_episodes):
        """
        Get bounds for current episode.

        Args:
            episode: Current episode number (0-indexed)
            total_episodes: Total number of training episodes

        Returns:
            tuple: (Q_min, Q_max) for current episode
        """
        # Linear interpolation based on training progress
        progress = episode / total_episodes

        Q_min = self.Q_min_init + progress * (self.Q_min_final - self.Q_min_init)
        Q_max = self.Q_max_init + progress * (self.Q_max_final - self.Q_max_init)

        return Q_min, Q_max

    def get_bounds_nonlinear(self, episode, total_episodes, schedule='quadratic'):
        """
        Get bounds with non-linear scheduling.

        Args:
            episode: Current episode number
            total_episodes: Total number of training episodes
            schedule: 'quadratic', 'exponential', or 'cosine'

        Returns:
            tuple: (Q_min, Q_max) for current episode
        """
        progress = episode / total_episodes

        if schedule == 'quadratic':
            # Slow at start, fast at end (more exploration early)
            alpha = progress ** 2
        elif schedule == 'exponential':
            # Very slow at start, very fast at end
            alpha = (np.exp(progress) - 1) / (np.e - 1)
        elif schedule == 'cosine':
            # Smooth transition
            alpha = (1 - np.cos(progress * np.pi)) / 2
        else:
            # Default to linear
            alpha = progress

        Q_min = self.Q_min_init + alpha * (self.Q_min_final - self.Q_min_init)
        Q_max = self.Q_max_init + alpha * (self.Q_max_final - self.Q_max_init)

        return Q_min, Q_max


class DelayedQBound:
    """
    Delayed Q-value bounds that activate after initial exploration phase.

    No bounds for first N episodes, then apply fixed bounds.

    Example:
        delayed = DelayedQBound(
            Q_min=-200, Q_max=0,
            delay_episodes=500
        )

        # Returns None, None before delay
        # Returns Q_min, Q_max after delay
        Q_min, Q_max = delayed.get_bounds(episode=100)
    """

    def __init__(self, Q_min, Q_max, delay_episodes):
        """
        Initialize delayed bounds.

        Args:
            Q_min: Lower bound (after delay)
            Q_max: Upper bound (after delay)
            delay_episodes: Number of episodes to wait before activating bounds
        """
        self.Q_min = Q_min
        self.Q_max = Q_max
        self.delay_episodes = delay_episodes

    def get_bounds(self, episode):
        """
        Get bounds for current episode.

        Args:
            episode: Current episode number (0-indexed)

        Returns:
            tuple: (Q_min, Q_max) or (None, None) if still in delay phase
        """
        if episode < self.delay_episodes:
            return None, None
        else:
            return self.Q_min, self.Q_max

    def is_active(self, episode):
        """Check if bounds are active for given episode."""
        return episode >= self.delay_episodes


class ProgressiveQBound:
    """
    Progressive bounds that start loose and progressively tighten.

    Combines delayed activation with adaptive tightening.

    Example:
        progressive = ProgressiveQBound(
            Q_min_init=-1000, Q_max_init=500,
            Q_min_final=-200, Q_max_final=0,
            delay_episodes=200,
            adaptation_episodes=800
        )
    """

    def __init__(self, Q_min_init, Q_max_init, Q_min_final, Q_max_final,
                 delay_episodes, adaptation_episodes):
        """
        Initialize progressive bounds.

        Args:
            Q_min_init: Initial lower bound (loose)
            Q_max_init: Initial upper bound (loose)
            Q_min_final: Final lower bound (tight)
            Q_max_final: Final upper bound (tight)
            delay_episodes: Episodes before bounds activate
            adaptation_episodes: Episodes over which to adapt from init to final
        """
        self.Q_min_init = Q_min_init
        self.Q_max_init = Q_max_init
        self.Q_min_final = Q_min_final
        self.Q_max_final = Q_max_final
        self.delay_episodes = delay_episodes
        self.adaptation_episodes = adaptation_episodes

    def get_bounds(self, episode):
        """
        Get bounds for current episode.

        Phase 1 (0 to delay_episodes): No bounds
        Phase 2 (delay to delay+adaptation): Linearly adapt from init to final
        Phase 3 (after delay+adaptation): Final bounds

        Args:
            episode: Current episode number (0-indexed)

        Returns:
            tuple: (Q_min, Q_max) or (None, None) in delay phase
        """
        if episode < self.delay_episodes:
            # Phase 1: No bounds
            return None, None

        elif episode < self.delay_episodes + self.adaptation_episodes:
            # Phase 2: Adaptive bounds
            adaptation_progress = (episode - self.delay_episodes) / self.adaptation_episodes
            Q_min = self.Q_min_init + adaptation_progress * (self.Q_min_final - self.Q_min_init)
            Q_max = self.Q_max_init + adaptation_progress * (self.Q_max_final - self.Q_max_init)
            return Q_min, Q_max

        else:
            # Phase 3: Final tight bounds
            return self.Q_min_final, self.Q_max_final


# Utility function for experiment scripts
def should_use_qbound(episode, exploration_episodes=200):
    """
    Simple helper for delayed QBound experiments.

    Args:
        episode: Current episode number
        exploration_episodes: Number of episodes before activating QBound

    Returns:
        bool: True if QBound should be active
    """
    return episode >= exploration_episodes
