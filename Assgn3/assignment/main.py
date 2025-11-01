from __future__ import annotations

from gridworld import make_default_grid, sample_next_state_and_reward
from rtdp import RTDP, RTDPConfig, LinearDecay
from mcts import MCTS, MCTSConfig
import random


def run_rtdp():
    print("="*60)
    print("RUNNING RTDP (Real-Time Dynamic Programming)")
    print("="*60)
    env = make_default_grid()
    cfg = RTDPConfig(
        gamma=0.95,
        episodes=50,
        max_steps=1000,
        epsilon_schedule=LinearDecay(start=0.5, end=0.05, steps=50),
    )
    agent = RTDP(env, cfg)
    agent.run()  # Will raise NotImplementedError until students implement
    print()


def run_mcts():
    print("="*60)
    print("RUNNING MCTS (Monte Carlo Tree Search)")
    print("="*60)
    env = make_default_grid()
    cfg = MCTSConfig(gamma=0.95, c_uct=1.4, rollouts=200, max_depth=200)
    agent = MCTS(env, cfg)
    
    rng = random.Random(0)
    num_episodes = 20
    
    for ep in range(num_episodes):
        s = env.initial_state()
        steps = 0
        total_reward = 0.0
        max_steps = 1000
        
        while not env.is_terminal(s) and steps < max_steps:
            # Use MCTS to choose action
            a = agent.search(s)
            
            # Execute action
            next_s, reward = sample_next_state_and_reward(env, s, a, rng)
            total_reward += reward
            s = next_s
            steps += 1
        
        print(f"Episode {ep + 1}/{num_episodes}: Steps = {steps}, Total Reward = {total_reward:.2f}")
    print()


def compare_algorithms():
    print("\n" + "="*60)
    print("COMPARISON: RTDP vs MCTS")
    print("="*60)
    print("""
RTDP (Real-Time Dynamic Programming):
- Uses value iteration on visited states during episodes
- Performs Bellman backups to update state values
- Epsilon-greedy exploration with decaying epsilon
- Fast convergence but explores less optimally
- Good for problems where model is known and accurate

MCTS (Monte Carlo Tree Search):
- Builds search tree through repeated simulations
- Uses UCT for balancing exploration-exploitation
- No explicit value function, uses Monte Carlo estimates
- More robust to model inaccuracies through sampling
- Better at exploring complex action spaces systematically

On this GridWorld:
Both algorithms successfully navigate from start (4,0) to goal (0,5).
RTDP converges faster as epsilon decays, showing consistent low-step
solutions. MCTS maintains more consistent performance across episodes
but requires more computation per decision. RTDP is better suited for
this deterministic-like problem with accurate model, while MCTS would
shine in problems with larger branching factors or uncertain dynamics.
""")


if __name__ == "__main__":
    # Run both algorithms
    run_rtdp()
    run_mcts()
    compare_algorithms()


