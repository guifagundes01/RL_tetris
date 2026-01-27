"""
Linear Q-Learning Agent using FeatureVectorObservation from GroupedActionsObservations.

Uses the pre-computed features from the wrapper, mapped to match the original
policy_linear_Q.py feature semantics.
"""
import numpy as np
import gymnasium as gym
from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations
from tetris_gymnasium.wrappers.observation import FeatureVectorObservation


class LinearQAgentGrouped:
    """
    Linear Q-learning agent that works with GroupedActionsObservations + FeatureVectorObservation.
    
    FeatureVectorObservation provides 13 features per action:
    [0-9]: column heights (actual heights, where higher = taller stack = worse)
    [10]: total holes count
    [11]: bumpiness (sum of height differences)
    [12]: total height (sum of column heights)
    
    Original policy_linear_Q.py uses different semantics:
    - heights(): returns "empty rows from top" (higher = more space = better)
    - So we need to INVERT the height interpretation
    """
    
    def __init__(self, learning_rate=0.01, discount=0.95, epsilon=1.0):
        # Weights for: [lines_cleared, height, holes, bumpiness]
        self.weights = np.zeros(4)
        
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        
        # Board parameters (standard Tetris board)
        self.board_height = 20  # rows
        self.board_width = 10   # columns
        
    
    def get_features_from_obs(self, obs_row, lines_cleared):
        """
        Convert FeatureVectorObservation features to match original policy_linear_Q.py semantics.
        
        FeatureVectorObservation output order (from source code):
        - indices 0-9: Column heights (actual height, higher = taller stack)
        - index 10: Max height
        - index 11: Holes count
        - index 12: Bumpiness
        
        Original heights() returns "empty rows from top" where:
        - Higher value = more empty space = better (less stacked)
        
        We convert by: empty_rows = board_height - actual_height
        """
        # Extract from FeatureVectorObservation format (CORRECT ORDER from source)
        col_heights_actual = obs_row[:10]  # Actual column heights (0 to ~20)
        max_height = obs_row[10]           # Max height
        total_holes = obs_row[11]          # Holes count
        bumpiness = obs_row[12]            # Bumpiness
        
        # Convert actual heights to "empty rows from top" to match original
        # Original: heights[i] = row index of first block ≈ empty rows from top
        # FeatureVector: col_heights[i] = actual height = board_height - empty_rows
        # So: empty_rows = board_height - actual_height
        col_heights_original = [self.board_height - h for h in col_heights_actual]
        
        # 1. Aggregate Height (average "empty rows from top")
        if len(col_heights_original) > 0:
            agg_height = sum(col_heights_original) / len(col_heights_original)
        else:
            agg_height = 0
        scaled_height = agg_height / 24.0  # Normalize (same as original)
        
        # 2. Number of Holes (use directly)
        scaled_holes = total_holes / 50.0
        
        # 3. Bumpiness (use directly)
        scaled_bumpiness = bumpiness / 216.0
        
        # 4. Lines Cleared
        scaled_lines = lines_cleared / 4.0
        
        return np.array([scaled_lines, scaled_height, scaled_holes, scaled_bumpiness])
    
    def evaluate(self, features):
        """Linear Q-value: Q(s) = weights · features"""
        return np.dot(self.weights, features)
    
    def choose_action(self, obs, action_mask, lines_cleared_per_action=None):
        """
        Choose the best action from grouped observations.
        
        Args:
            obs: shape (n_actions, 13) - feature vectors for each possible placement
            action_mask: shape (n_actions,) - 1 if action is valid, 0 otherwise
            lines_cleared_per_action: optional array of lines cleared per action
        
        Returns:
            best_action: int - the action index to take
            best_features: the features of the chosen afterstate
        """
        valid_actions = np.where(action_mask == 1)[0]
        
        if len(valid_actions) == 0:
            return None, None
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            action = np.random.choice(valid_actions)
            lines = lines_cleared_per_action[action] if lines_cleared_per_action is not None else 0
            features = self.get_features_from_obs(obs[action], lines)
            return action, features
        
        # Greedy: evaluate all valid actions
        best_score = -float('inf')
        best_action = None
        best_features = None
        
        for action in valid_actions:
            lines = lines_cleared_per_action[action] if lines_cleared_per_action is not None else 0
            features = self.get_features_from_obs(obs[action], lines)
            score = self.evaluate(features)
            
            if score > best_score:
                best_score = score
                best_action = action
                best_features = features
        
        return best_action, best_features
    
    def learn(self, features, reward, next_best_q):
        """Q-learning update with linear function approximation."""
        target = reward + self.gamma * next_best_q
        prediction = self.evaluate(features)
        error = target - prediction
        self.weights += self.lr * error * features
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


def create_grouped_env(env_id="tetris_gymnasium/Tetris", render_mode=None, **kwargs):
    """
    Create environment with GroupedActionsObservations + FeatureVectorObservation wrapper.
    
    FeatureVectorObservation output order (from source code):
    - indices 0-9: Column heights (report_height)
    - index 10: Max height (report_max_height)
    - index 11: Number of holes (report_holes)
    - index 12: Bumpiness (report_bumpiness)
    
    Total: 13 features per action when all enabled.
    """
    env = gym.make(env_id, render_mode=render_mode, gravity=False, **kwargs)
    
    # Configure FeatureVectorObservation with explicit feature flags
    feature_wrapper = FeatureVectorObservation(
        env,
        report_height=True,      # indices 0-9: column heights
        report_max_height=True,  # index 10: max height
        report_holes=True,       # index 11: holes count
        report_bumpiness=True    # index 12: bumpiness
    )
    
    env = GroupedActionsObservations(
        env,
        observation_wrappers=[feature_wrapper]
    )
    return env
