import numpy as np
from algorithms.base_algorithm import BaseMABAlgorithm

class UCB(BaseMABAlgorithm):
    """
    Upper Confidence Bound (UCB) algorithm
    Balances exploration and exploitation using confidence bounds
    """
    def __init__(self, n_arms: int, c: float = 2.0, **kwargs):
        super().__init__(n_arms, **kwargs)
        self.c = c  # Exploration parameter
        
    def select_arm(self) -> int:
        """
        ## IMPLEMENT UCB ALGORITHM HERE ##
        
        Input: None (uses self.estimates, self.pulls, self.c)
        Output: int - selected arm index
        
        Strategy: UCB balances exploration and exploitation using confidence bounds
        - If some arms haven't been pulled: pull one of them
        - Otherwise: select arm with highest UCB value
        - UCB formula: estimate + c * sqrt(log(total_pulls) / arm_pulls)
        """
        # Check for unpulled arms
        unpulled_arms = np.where(self.pulls == 0)[0]
        
        # If there are unpulled arms, return the first one
        if len(unpulled_arms) > 0:
            return unpulled_arms[0]
        
        # Calculate total pulls
        total_pulls = np.sum(self.pulls)
        
        # Calculate UCB values for all arms
        ucb_values = self.estimates + self.c * np.sqrt(np.log(total_pulls) / self.pulls)
        
        # Return arm with highest UCB value
        return np.argmax(ucb_values)
