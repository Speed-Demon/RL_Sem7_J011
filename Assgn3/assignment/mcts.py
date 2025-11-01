from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from gridworld import MDP, State, Action, sample_next_state_and_reward


@dataclass
class MCTSConfig:
    gamma: float = 0.95
    c_uct: float = 1.4
    rollouts: int = 200
    max_depth: int = 200


class Node:
    def __init__(self, state: State, parent: Optional[Tuple["Node", Action]] = None) -> None:
        self.state = state
        self.parent = parent
        self.children: Dict[Action, Node] = {}
        self.visits = 0
        self.value_sum = 0.0

    @property
    def q(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / float(self.visits)


class MCTS:
    def __init__(self, mdp: MDP, cfg: MCTSConfig, rng=None, heuristic=None) -> None:
        self.mdp = mdp
        self.cfg = cfg
        self.rng = rng
        self.heuristic = heuristic
        if self.rng is None:
            import random

            self.rng = random.Random(0)

    def search(self, root_state: State) -> Action:
        root = Node(root_state)
        for _ in range(self.cfg.rollouts):
            # YOUR CODE HERE: one MCTS iteration (selection, expansion, rollout, backprop)
            raise NotImplementedError

        # choose action with most visits
        best_a = None
        best_v = -1
        for a, ch in root.children.items():
            if ch.visits > best_v:
                best_v = ch.visits
                best_a = a
        if best_a is None:
            actions = list(self.mdp.actions(root_state))
            if not actions:
                raise RuntimeError("MCTS on terminal state")
            best_a = actions[0]
        return best_a

