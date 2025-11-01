from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from gridworld import MDP, State, Action, sample_next_state_and_reward


@dataclass
class LinearDecay:
    start: float
    end: float
    steps: int

    def value(self, t: int) -> float:
        if t <= 0:
            return float(self.start)
        if t >= self.steps:
            return float(self.end)
        frac = t / float(self.steps)
        return float(self.start + frac * (self.end - self.start))


@dataclass
class RTDPConfig:
    gamma: float = 0.95
    episodes: int = 50
    max_steps: int = 1_000
    epsilon_schedule: LinearDecay | None = None


class RTDP:
    def __init__(self, mdp: MDP, cfg: RTDPConfig, rng=None, heuristic=None) -> None:
        self.mdp = mdp
        self.cfg = cfg
        self.rng = rng
        self.heuristic = heuristic
        self.V: Dict[State, float] = {}

        if self.rng is None:
            import random

            self.rng = random.Random(0)

    def value(self, s: State) -> float:
        if s not in self.V:
            self.V[s] = float(self.heuristic(s) if self.heuristic else 0.0)
        return self.V[s]

    def bellman_backup(self, s: State) -> float:
        actions = self.mdp.actions(s)
        if not actions:
            self.V[s] = 0.0
            return 0.0

        # YOUR CODE HERE: compute V(s) = max_a E[r + gamma * V(s')]
        raise NotImplementedError

    def select_action(self, s: State, epsilon: float) -> Action:
        actions = list(self.mdp.actions(s))
        assert actions
        # YOUR CODE HERE: epsilon-greedy over one-step lookahead Q(s,a)
        # sample a random action with prob epsilon, otherwise pick argmax Q(s,a)
        raise NotImplementedError

    def run(self) -> None:
        episodes = self.cfg.episodes
        for ep in range(episodes):
            s = self.mdp.initial_state()
            steps = 0
            epsilon = self.cfg.epsilon_schedule.value(ep) if self.cfg.epsilon_schedule else 0.0

            # YOUR CODE HERE: RTDP episode loop
            # while not terminal and steps < max_steps:
            #   - bellman_backup(s)
            #   - a = select_action(s, epsilon)
            #   - s, r = sample_next_state_and_reward(...)
            #   - steps += 1
            raise NotImplementedError

