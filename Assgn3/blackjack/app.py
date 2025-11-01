import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import random
from collections import defaultdict, Counter
import json
import time

# Page config
st.set_page_config(
    page_title="Blackjack",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'game_state' not in st.session_state:
    st.session_state.game_state = {
        'current_team': 'A',
        'current_round': 1,
        'current_state': 0,
        'round_history': [],
        'team_scores': {'A': 0, 'B': 0},
        'transition_counts': defaultdict(int),
        'state_frequencies': defaultdict(int),
        'game_history': [],
        'current_round_cards': [],
        'team_names': {'A': 'Team A', 'B': 'Team B'},
        'show_confetti': False
    }

def draw_card():
    """Draw a card from [2, 3, 4, 5, 6] with uniform probability"""
    return random.choice([2, 3, 4, 5, 6])

def hit_action(current_state):
    """Perform hit action and return new state"""
    card = draw_card()
    new_state = current_state + card
    return new_state, card

def stand_action(current_state):
    """Perform stand action and return reward"""
    return current_state

def update_transition_counts(old_state, new_state):
    """Update transition count for Markov chain visualization"""
    st.session_state.game_state['transition_counts'][(old_state, new_state)] += 1

def create_progress_bar(current_state):
    """Create a progress bar showing how close to 21 you are"""
    if current_state > 21:
        progress = 100
        color = "red"
    else:
        progress = (current_state / 21) * 100
        if current_state >= 18:
            color = "green"
        elif current_state >= 15:
            color = "orange"
        else:
            color = "blue"
    
    st.markdown(f"""
    <div style="margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span>0</span>
            <span>21</span>
        </div>
        <div style="background-color: #f0f0f0; border-radius: 10px; height: 20px; overflow: hidden;">
            <div style="background-color: {color}; height: 100%; width: {progress}%; transition: width 0.3s ease;"></div>
        </div>
        <div style="text-align: center; margin-top: 5px; font-weight: bold; color: {color};">
            {current_state}/21
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_confetti():
    """Show confetti animation for perfect rounds"""
    confetti_html = """
    <div id="confetti-container" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 9999;">
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 48px; animation: bounce 2s ease-in-out;">
            🎉🎊🎈
        </div>
    </div>
    <style>
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {transform: translate(-50%, -50%) scale(1);}
        40% {transform: translate(-50%, -50%) scale(1.2);}
        60% {transform: translate(-50%, -50%) scale(0.9);}
    }
    </style>
    <script>
    setTimeout(function() {
        document.getElementById('confetti-container').style.display = 'none';
    }, 3000);
    </script>
    """
    st.markdown(confetti_html, unsafe_allow_html=True)

def create_simple_state_plot():
    """Create simple bar plot of state frequencies"""
    frequencies = st.session_state.game_state['state_frequencies']
    
    if not frequencies:
        return go.Figure()
    
    states = sorted(frequencies.keys())
    counts = [frequencies[state] for state in states]
    
    fig = go.Figure(data=[
        go.Bar(x=states, y=counts, marker_color='lightblue')
    ])
    
    fig.update_layout(
        title="State Visits",
        xaxis_title="State",
        yaxis_title="Count",
        height=300
    )
    
    return fig

def create_simple_reward_plot():
    """Create simple reward plot per team"""
    game_history = st.session_state.game_state['game_history']
    
    if not game_history:
        return go.Figure()
    
    team_a_rewards = [round_data['reward'] for round_data in game_history if round_data['team'] == 'A']
    team_b_rewards = [round_data['reward'] for round_data in game_history if round_data['team'] == 'B']
    
    fig = go.Figure()
    
    if team_a_rewards:
        fig.add_trace(go.Scatter(y=team_a_rewards, name='Team A', mode='lines+markers', line=dict(color='lightblue')))
    if team_b_rewards:
        fig.add_trace(go.Scatter(y=team_b_rewards, name='Team B', mode='lines+markers', line=dict(color='lightcoral')))
    
    fig.update_layout(
        title="Team Scores Over Time",
        yaxis_title="Reward",
        height=300
    )
    
    return fig

# Main app
def main():
    st.title("🃏 Blackjack")
    
    # Team name selection
    if 'team_names_set' not in st.session_state:
        st.session_state.team_names_set = False
    
    if not st.session_state.team_names_set:
        # Ensure team_names exists in game_state
        if 'team_names' not in st.session_state.game_state:
            st.session_state.game_state['team_names'] = {'A': 'Team A', 'B': 'Team B'}
        
        st.subheader("🏆 Set Your Team Names")
        col1, col2 = st.columns(2)
        
        with col1:
            team_a_name = st.text_input("Team A Name:", value="Team A", key="team_a_input")
        with col2:
            team_b_name = st.text_input("Team B Name:", value="Team B", key="team_b_input")
        
        if st.button("Start Game!"):
            st.session_state.game_state['team_names']['A'] = team_a_name
            st.session_state.game_state['team_names']['B'] = team_b_name
            st.session_state.team_names_set = True
            st.rerun()
        return
    
    # Show confetti if needed
    if st.session_state.game_state.get('show_confetti', False):
        show_confetti()
        st.session_state.game_state['show_confetti'] = False
    
    # Sidebar - minimal
    with st.sidebar:
        st.header("📚 Concepts")
        st.markdown("**Markov Property**: Next state depends only on current state")
        st.markdown("**States**: Hand totals (0-21+)")
        st.markdown("**Cards**: [2,3,4,5,6] uniform")
        st.markdown("**Bust**: >21 = 0 reward")
        
        # Team names display
        st.markdown("---")
        st.header("🏆 Teams")
        team_names = st.session_state.game_state.get('team_names', {'A': 'Team A', 'B': 'Team B'})
        for team, name in team_names.items():
            team_icon = "🔴" if team == 'B' else "🔵"
            st.markdown(f"{team_icon} {name}")
        
        if st.button("🔄 Reset"):
            st.session_state.game_state = {
                'current_team': 'A',
                'current_round': 1,
                'current_state': 0,
                'round_history': [],
                'team_scores': {'A': 0, 'B': 0},
                'transition_counts': defaultdict(int),
                'state_frequencies': defaultdict(int),
                'game_history': [],
                'current_round_cards': [],
                'team_names': {'A': 'Team A', 'B': 'Team B'},
                'show_confetti': False
            }
            st.session_state.team_names_set = False
            st.rerun()
    
    # Game interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        current_team = st.session_state.game_state['current_team']
        team_names = st.session_state.game_state.get('team_names', {'A': 'Team A', 'B': 'Team B'})
        team_name = team_names[current_team]
        team_color = "🔴" if current_team == 'B' else "🔵"
        st.subheader(f"{team_color} {team_name} - Round {st.session_state.game_state['current_round']}")
        
        # Current state display
        current_state = st.session_state.game_state['current_state']
        st.metric("Total", current_state)
        
        # Progress bar
        create_progress_bar(current_state)
        
        # Show current round cards
        if st.session_state.game_state['current_round_cards']:
            cards_str = " + ".join(map(str, st.session_state.game_state['current_round_cards']))
            st.markdown(f"<div style='font-size: 24px; font-weight: bold; color: #1f77b4;'><strong>Cards this round:</strong> {cards_str}</div>", unsafe_allow_html=True)
        
        # Game buttons
        col_hit, col_stand = st.columns(2)
        
        with col_hit:
            if st.button("Hit", use_container_width=True):
                new_state, card = hit_action(current_state)
                st.session_state.game_state['current_state'] = new_state
                st.session_state.game_state['state_frequencies'][new_state] += 1
                st.session_state.game_state['current_round_cards'].append(card)
                update_transition_counts(current_state, new_state)
                
                if new_state > 21:  # Bust
                    st.error(f"Bust! {card} → {new_state}")
                    # End round
                    round_data = {
                        'team': st.session_state.game_state['current_team'],
                        'round': st.session_state.game_state['current_round'],
                        'final_state': new_state,
                        'reward': 0,
                        'result': 'Bust',
                        'cards': st.session_state.game_state['current_round_cards'].copy()
                    }
                    st.session_state.game_state['game_history'].append(round_data)
                    st.session_state.game_state['team_scores'][st.session_state.game_state['current_team']] += 0
                    
                    # Switch teams or rounds
                    if st.session_state.game_state['current_round'] == 10:
                        if st.session_state.game_state['current_team'] == 'A':
                            st.session_state.game_state['current_team'] = 'B'
                            st.session_state.game_state['current_round'] = 1
                        else:
                            st.success("Game Complete!")
                    else:
                        st.session_state.game_state['current_round'] += 1
                    
                    st.session_state.game_state['current_state'] = 0
                    st.session_state.game_state['current_round_cards'] = []
                    st.rerun()
                else:
                    st.success(f"Drew {card} → {new_state}")
                    st.rerun()
        
        with col_stand:
            if st.button("Stand", use_container_width=True):
                reward = stand_action(current_state)
                
                # Check for perfect round and trigger confetti
                if reward == 21:
                    st.session_state.game_state['show_confetti'] = True
                    st.success(f"🎯 PERFECT! {current_state} = {reward}")
                elif reward >= 20:
                    st.success(f"🔥 Excellent! {current_state} = {reward}")
                elif reward >= 18:
                    st.success(f"👍 Good! {current_state} = {reward}")
                else:
                    st.success(f"Stand: {current_state} = {reward}")
                
                # End round
                round_data = {
                    'team': st.session_state.game_state['current_team'],
                    'round': st.session_state.game_state['current_round'],
                    'final_state': current_state,
                    'reward': reward,
                    'result': 'Stand',
                    'cards': st.session_state.game_state['current_round_cards'].copy()
                }
                st.session_state.game_state['game_history'].append(round_data)
                st.session_state.game_state['team_scores'][st.session_state.game_state['current_team']] += reward
                
                # Switch teams or rounds
                if st.session_state.game_state['current_round'] == 10:
                    if st.session_state.game_state['current_team'] == 'A':
                        st.session_state.game_state['current_team'] = 'B'
                        st.session_state.game_state['current_round'] = 1
                    else:
                        st.success("Game Complete!")
                else:
                    st.session_state.game_state['current_round'] += 1
                
                st.session_state.game_state['current_state'] = 0
                st.session_state.game_state['current_round_cards'] = []
                st.rerun()
    
    with col2:
        st.subheader("Scores")
        team_names = st.session_state.game_state.get('team_names', {'A': 'Team A', 'B': 'Team B'})
        for team, score in st.session_state.game_state['team_scores'].items():
            team_icon = "🔴" if team == 'B' else "🔵"
            team_name = team_names[team]
            st.metric(f"{team_icon} {team_name}", score)
        
        st.markdown("---")
        st.subheader("History")
        if st.session_state.game_state['game_history']:
            recent_rounds = st.session_state.game_state['game_history'][-5:]
            for round_data in recent_rounds:
                cards_str = " + ".join(map(str, round_data['cards'])) if round_data['cards'] else "0"
                team_icon = "🔴" if round_data['team'] == 'B' else "🔵"
                team_names = st.session_state.game_state.get('team_names', {'A': 'Team A', 'B': 'Team B'})
                team_name = team_names[round_data['team']]
                
                # Add fun emojis based on result
                result_emoji = "💥" if round_data['result'] == 'Bust' else "✅"
                if round_data['reward'] == 21:
                    result_emoji = "🎯"
                elif round_data['reward'] >= 20:
                    result_emoji = "🔥"
                
                st.markdown(f"<div style='font-size: 18px;'>{team_icon} {result_emoji} <strong>{team_name} R{round_data['round']}</strong>: {round_data['result']} ({round_data['reward']}) - Cards: {cards_str}</div>", unsafe_allow_html=True)
    
    # Simple analytics section
    st.markdown("---")
    st.header("Analytics")
    
    col_analytics1, col_analytics2 = st.columns(2)
    
    with col_analytics1:
        st.plotly_chart(create_simple_state_plot(), use_container_width=True)
    
    with col_analytics2:
        st.plotly_chart(create_simple_reward_plot(), use_container_width=True)

if __name__ == "__main__":
    main() 