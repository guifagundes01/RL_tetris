import numpy as np
import random
from policies import get_possible_next_states, heights, holes # reusing your code

class LinearQLearningAgent:
    def __init__(self, num_features=4, learning_rate=0.01, discount=0.95, epsilon=1.0):
        # 1. Initialize weights randomly (or to 0)
        # Expectation: 'holes' weight should become negative over time.
        self.weights = np.zeros(num_features)
        # self.weights = np.array([0.5, -0.01, -0.05, -0.05])
        
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01

    def get_features(self, board, lines_cleared):
        """
        Extracts high-level features and NORMALIZES them.
        """
        # 1. Aggregate Height
        # Max possible height sum is 24 (rows) * 18 (cols) = 432
        col_heights = heights(board)
        agg_height = sum(col_heights) / len(col_heights)
        scaled_height = agg_height / 24.0
        
        # 2. Number of Holes
        # A very bad board might have ~50 holes. 
        num_holes = holes(board)
        scaled_holes = num_holes / 50.0 
        
        # 3. Bumpiness
        # Max bumpiness is roughly 24 * 17 transitions = 216
        bumpiness = 0
        for i in range(len(col_heights) - 1):
            bumpiness += abs(col_heights[i] - col_heights[i+1])
        scaled_bumpiness = bumpiness / 216.0
        
        # 4. Lines Cleared
        # Max lines in one turn is 4.
        scaled_lines = lines_cleared / 4.0

        return np.array([scaled_lines, scaled_height, scaled_holes, scaled_bumpiness])

    def evaluate_board(self, features):
        """
        Computes the Q-value (Score) using the linear equation:
        Q(s, a) = weights * features
        """
        return np.dot(self.weights, features)

    def choose_action(self, env):
        """
        Selects the best sequence of moves (Grouped Action).
        """
        # 1. Get all possible future states (Grouped Actions)
        candidates = get_possible_next_states(env)
        
        if not candidates:
            return None # Game Over likely

        # 2. Exploration (Epsilon-Greedy)
        # Sometimes pick a random move to see if it yields a better reward
        # if random.random() < self.epsilon:
        #     return random.choice(candidates)

        # 3. Exploitation (Greedy)
        best_score = -float('inf')
        best_candidate = None
        
        for cand in candidates:
            lines_cleared = cand["lines_cleared"]
            feats = self.get_features(cand["board"], lines_cleared)
            score = self.evaluate_board(feats)

            if score > best_score:
                best_score = score
                best_candidate = cand

        return best_candidate

    def learn(self, old_features, reward, new_state_best_q):
        """
        The Learning Step (Gradient Descent).
        Update weights based on the difference between expected and actual reward.
        """
        # Q-Learning Target: Reward + Discount * Max_Future_Q
        target = reward + self.gamma * new_state_best_q
        
        # Current Prediction
        prediction = self.evaluate_board(old_features)
        
        # The Error (TD-Error)
        error = target - prediction
        
        # Update Weights: w = w + alpha * error * features
        # If error is positive (surprise reward), increase weights for features present.
        self.weights += self.lr * error * old_features
