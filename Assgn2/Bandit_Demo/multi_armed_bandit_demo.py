import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
import random

# Set page config
st.set_page_config(
    page_title="Multi-Armed Bandit Demo",
    page_icon="🎰",
    layout="wide"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .game-stats {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .lever-button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 1rem;
        border-radius: 10px;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 0.5rem;
    }
    .metric-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        text-align: center;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'game_state' not in st.session_state:
    st.session_state.game_state = {
        'pulls_left': 20,
        'total_reward': 0,
        'lever_pulls': defaultdict(int),
        'lever_rewards': defaultdict(int),
        'game_history': [],
        'true_probabilities': {},
        'game_started': False,
        'game_finished': False
    }

def initialize_game():
    """Initialize the game with random true probabilities"""
    st.session_state.game_state = {
        'pulls_left': 20,
        'total_reward': 0,
        'lever_pulls': defaultdict(int),
        'lever_rewards': defaultdict(int),
        'game_history': [],
        'true_probabilities': {
            'Lever A': random.uniform(0.3, 0.7),
            'Lever B': random.uniform(0.3, 0.7),
            'Lever C': random.uniform(0.3, 0.7),
            'Lever D': random.uniform(0.3, 0.7),
            'Lever E': random.uniform(0.3, 0.7)
        },
        'game_started': False,
        'game_finished': False
    }

def pull_lever(lever_name):
    """Pull a lever and return the result"""
    if st.session_state.game_state['pulls_left'] <= 0:
        return False
    
    # Get true probability for this lever
    true_prob = st.session_state.game_state['true_probabilities'][lever_name]
    
    # Simulate the pull
    reward = 1 if random.random() < true_prob else 0
    
    # Update game state
    st.session_state.game_state['pulls_left'] -= 1
    st.session_state.game_state['total_reward'] += reward
    st.session_state.game_state['lever_pulls'][lever_name] += 1
    st.session_state.game_state['lever_rewards'][lever_name] += reward
    
    # Record the action
    st.session_state.game_state['game_history'].append({
        'pull': 20 - st.session_state.game_state['pulls_left'],
        'lever': lever_name,
        'reward': reward,
        'total_reward': st.session_state.game_state['total_reward']
    })
    
    return reward

def create_lever_interface():
    """Create the lever interface with better aesthetics"""
    st.markdown("### 🎰 Choose Your Lever")
    st.markdown("Click any lever to pull it and see the result!")
    
    # Create lever buttons in a more spaced layout
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("🎰 A", key="lever_a", use_container_width=True, help="Pull Lever A"):
            reward = pull_lever("Lever A")
            st.rerun()
    
    with col2:
        if st.button("🎰 B", key="lever_b", use_container_width=True, help="Pull Lever B"):
            reward = pull_lever("Lever B")
            st.rerun()
    
    with col3:
        if st.button("🎰 C", key="lever_c", use_container_width=True, help="Pull Lever C"):
            reward = pull_lever("Lever C")
            st.rerun()
    
    with col4:
        if st.button("🎰 D", key="lever_d", use_container_width=True, help="Pull Lever D"):
            reward = pull_lever("Lever D")
            st.rerun()
    
    with col5:
        if st.button("🎰 E", key="lever_e", use_container_width=True, help="Pull Lever E"):
            reward = pull_lever("Lever E")
            st.rerun()

def display_game_stats():
    """Display game statistics with better layout"""
    st.markdown("### 📊 Game Statistics")
    
    # Main metrics in a clean row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Pulls Left", st.session_state.game_state['pulls_left'])
    
    with col2:
        st.metric("Total Reward", st.session_state.game_state['total_reward'])
    
    with col3:
        if st.session_state.game_state['pulls_left'] < 20:
            avg_reward = st.session_state.game_state['total_reward'] / (20 - st.session_state.game_state['pulls_left'])
            st.metric("Average Reward", f"{avg_reward:.2f}")
        else:
            st.metric("Average Reward", "0.00")
    
    with col4:
        if st.session_state.game_state['pulls_left'] == 10:
            st.info("🎯 **Mid-Game Strategy Check!**")
    
    # Lever performance table
    st.markdown("#### Lever Performance")
    lever_stats = []
    for lever in ['Lever A', 'Lever B', 'Lever C', 'Lever D', 'Lever E']:
        pulls = st.session_state.game_state['lever_pulls'][lever]
        rewards = st.session_state.game_state['lever_rewards'][lever]
        success_rate = rewards / pulls if pulls > 0 else 0
        lever_stats.append({
            'Lever': lever,
            'Pulls': pulls,
            'Rewards': rewards,
            'Success Rate': f"{success_rate:.2f}" if pulls > 0 else "0.00"
        })
    
    df_stats = pd.DataFrame(lever_stats)
    st.dataframe(df_stats, use_container_width=True, height=200)

def display_game_progress():
    """Display game progress with visual elements"""
    if not st.session_state.game_state['game_history']:
        return
    
    st.markdown("### 📈 Game Progress")
    
    # Create progress visualization
    history_df = pd.DataFrame(st.session_state.game_state['game_history'])
    
    # Progress bar
    progress = (20 - st.session_state.game_state['pulls_left']) / 20
    st.progress(progress)
    st.caption(f"Progress: {20 - st.session_state.game_state['pulls_left']}/20 pulls")
    
    # Reward chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_df['pull'],
        y=history_df['total_reward'],
        mode='lines+markers',
        name='Cumulative Reward',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8, color='#1f77b4')
    ))
    
    fig.update_layout(
        title="Cumulative Reward Over Time",
        xaxis_title="Pull Number",
        yaxis_title="Total Reward",
        height=300,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_final_results():
    """Show final results with better presentation"""
    st.markdown("### 🎯 Final Results")
    
    # Final score
    st.success(f"🎉 **Final Score: {st.session_state.game_state['total_reward']} points**")
    
    # True vs observed probabilities
    true_probs = st.session_state.game_state['true_probabilities']
    lever_names = list(true_probs.keys())
    true_values = [true_probs[lever] for lever in lever_names]
    observed_values = [st.session_state.game_state['lever_rewards'][lever] / 
                      max(st.session_state.game_state['lever_pulls'][lever], 1) 
                      for lever in lever_names]
    
    # Create comparison chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=lever_names,
        y=true_values,
        name='True Probability',
        marker_color='#2ca02c'
    ))
    fig.add_trace(go.Bar(
        x=lever_names,
        y=observed_values,
        name='Observed Success Rate',
        marker_color='#ff7f0e'
    ))
    
    fig.update_layout(
        title="True vs Observed Probabilities",
        xaxis_title="Lever",
        yaxis_title="Probability",
        barmode='group',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    best_true = max(true_probs.items(), key=lambda x: x[1])
    best_observed = max([(lever, st.session_state.game_state['lever_rewards'][lever] / 
                         max(st.session_state.game_state['lever_pulls'][lever], 1)) 
                        for lever in lever_names], key=lambda x: x[1])
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Best True Probability**: {best_true[0]} ({best_true[1]:.2f})")
    with col2:
        st.info(f"**Best Observed Rate**: {best_observed[0]} ({best_observed[1]:.2f})")
    
    if best_true[0] != best_observed[0]:
        st.warning("⚠️ **Key Insight**: The lever with the best true probability was NOT the one with the best observed success rate!")
        st.write("This demonstrates the exploration vs exploitation dilemma - we might abandon a potentially better option due to unlucky early results.")

def main():
    # Introduction
    if not st.session_state.game_state['game_started']:
        st.markdown("---")
        st.markdown("### 🍽️ The Restaurant Dilemma")
        st.markdown("""
        Imagine you're visiting a new city for 5 nights. On your first night, you find an amazing restaurant! 
        
        **The Dilemma**: Do you return to that great restaurant for the next 4 nights (exploit what you know is good), 
        or do you try new restaurants that might be even better (explore unknown options)?
        
        This is the **Exploration vs. Exploitation** tradeoff - the core challenge in reinforcement learning!
        """)
        
        st.markdown("### 🎮 Let's Play!")
        st.markdown("We have 5 levers (slot machines). Each has a secret probability of giving you 1 point. Your goal: maximize your score in 20 pulls!")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎮 Start the Game", type="primary", use_container_width=True):
                initialize_game()
                st.session_state.game_state['game_started'] = True
                st.rerun()
    
    else:
        # Game interface
        if st.session_state.game_state['pulls_left'] > 0:
            # Game status
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Pulls Left", st.session_state.game_state['pulls_left'])
            with col2:
                st.metric("Total Reward", st.session_state.game_state['total_reward'])
            
            # Lever interface
            st.markdown("---")
            create_lever_interface()
            
            # Game statistics and progress
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                display_game_stats()
            
            with col2:
                display_game_progress()
        
        else:
            # Game finished
            st.markdown("---")
            show_final_results()
            
            st.markdown("---")
            st.markdown("### 🔄 Play Again")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔄 New Game", type="primary", use_container_width=True):
                    initialize_game()
                    st.session_state.game_state['game_started'] = True
                    st.rerun()

if __name__ == "__main__":
    main() 