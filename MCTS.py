import numpy as np
import random
import chess

class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board.copy()
        self.parent = parent
        self.move = move  # The move that led to this node
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_moves = list(board.legal_moves)  # Moves yet to be explored
    
    def is_fully_expanded(self):
        return len(self.untried_moves) == 0
    
    def best_child(self, exploration_weight=1.4):
        """Selects the best child based on UCT (Upper Confidence Bound for Trees) using NumPy."""
        values = np.array([(child.value / (child.visits + 1e-6)) +
                          exploration_weight * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-6))
                          for child in self.children])
        return self.children[np.argmax(values)]

    def select_child(self):
        """Select the child with the highest UCT value."""
        return self.best_child()
    
    def expand(self):
        """Expands a new child from an untried move."""
        move = self.untried_moves.pop()
        new_board = self.board.copy()
        new_board.push(move)
        child_node = MCTSNode(new_board, parent=self, move=move)
        self.children.append(child_node)
        return child_node
    
    def rollout(self):
        """Performs a random rollout from the current state and returns the final result."""
        rollout_board = self.board.copy()
        while not rollout_board.is_game_over():
            legal_moves = list(rollout_board.legal_moves)
            rollout_board.push(random.choice(legal_moves))
        outcome = rollout_board.result()
        if outcome == "1-0":
            return 1 if self.board.turn == chess.WHITE else -1
        elif outcome == "0-1":
            return -1 if self.board.turn == chess.WHITE else 1
        return 0  # Draw

    def backpropagate(self, result):
        """Backpropagates the rollout result up the tree."""
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backpropagate(-result)  # Invert value for opponent's turn


def mcts(board, simulations=100):
    """Performs MCTS and returns the best move."""
    root = MCTSNode(board)
    
    for _ in range(simulations):
        node = root
        # Selection
        while node.is_fully_expanded() and node.children:
            node = node.select_child()
        # Expansion
        if not node.is_fully_expanded():
            node = node.expand()
        # Rollout
        result = node.rollout()
        # Backpropagation
        node.backpropagate(result)
    
    return root.best_child(exploration_weight=0).move  # Select best move without exploration