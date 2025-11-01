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
            # Selection: traverse tree using UCT
            node = root
            path = [node]
            depth = 0
            
            while node.children and not self.mdp.is_terminal(node.state) and depth < self.cfg.max_depth:
                # Select child using UCT
                action = self._select_uct(node)
                node = node.children[action]
                path.append(node)
                depth += 1
            
            # Expansion: if not terminal and not at max depth, expand
            if not self.mdp.is_terminal(node.state) and depth < self.cfg.max_depth:
                actions = list(self.mdp.actions(node.state))
                if actions:
                    # Pick an unexplored action
                    unexplored = [a for a in actions if a not in node.children]
                    if unexplored:
                        action = self.rng.choice(unexplored)
                        next_s, _ = sample_next_state_and_reward(self.mdp, node.state, action, self.rng)
                        child = Node(next_s, parent=(node, action))
                        node.children[action] = child
                        path.append(child)
                        node = child
                        depth += 1
            
            # Rollout: simulate from current node to terminal or max depth
            value = self._rollout(node.state, depth)
            
            # Backpropagation: update all nodes in path
            for n in path:
                n.visits += 1
                n.value_sum += value

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

    def _select_uct(self, node: Node) -> Action:
        """Select action using UCT formula"""
        best_action = None
        best_score = float('-inf')
        
        for action, child in node.children.items():
            if child.visits == 0:
                return action  # Prefer unvisited children
            
            # UCT formula: Q + c * sqrt(ln(N) / N_a)
            exploit = child.q
            explore = self.cfg.c_uct * math.sqrt(math.log(node.visits) / child.visits)
            score = exploit + explore
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def _rollout(self, state: State, depth: int) -> float:
        """Simulate random policy from state until terminal or max depth"""
        s = state
        total_return = 0.0
        discount = 1.0
        
        while not self.mdp.is_terminal(s) and depth < self.cfg.max_depth:
            actions = list(self.mdp.actions(s))
            if not actions:
                break
            
            action = self.rng.choice(actions)
            next_s, reward = sample_next_state_and_reward(self.mdp, s, action, self.rng)
            total_return += discount * reward
            discount *= self.cfg.gamma
            s = next_s
            depth += 1
        
        return total_return

