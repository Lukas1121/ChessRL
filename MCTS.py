import numpy as np
import random
import chess

class MCTSNode:
    def __init__(self, board, network, action_space, parent=None, move=None):
        self.board = board.copy()
        self.network = network
        self.action_space = action_space  # Pass the action space here
        self.parent = parent
        self.move = move  # The move that led to this node
        self.children = {}  # Map moves to child nodes
        self.visits = 0
        self.value_sum = 0.0
        self.is_terminal = board.is_game_over()

        if not self.is_terminal:
            board_tensor = network.board_to_tensor(self.board)
            policy, value = network.forward(board_tensor)
            legal_moves = list(self.board.legal_moves)
            self.priors = {}
            for move in legal_moves:
                action_index = self.action_space.index(move)
                self.priors[move] = policy[0, action_index].item()
        else:
            self.priors = {}

    def is_fully_expanded(self):
        """A node is fully expanded if all legal moves have been added as children."""
        return len(self.children) == len(list(self.board.legal_moves))

    def best_child(self, c_puct=1.4):
        """Selects the best child using a PUCT formula that incorporates prior probabilities."""
        best_score = -float('inf')
        best_child = None
        for move, child in self.children.items():
            # Q: average value; U: exploration bonus
            Q = child.value_sum / (child.visits + 1e-8)
            U = c_puct * self.priors[move] * np.sqrt(self.visits) / (1 + child.visits)
            score = Q + U
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self):
        """Expands one child not already in the tree."""
        legal_moves = list(self.board.legal_moves)
        unexpanded_moves = [move for move in legal_moves if move not in self.children]
        if not unexpanded_moves:
            return None
        move = random.choice(unexpanded_moves)
        new_board = self.board.copy()
        new_board.push(move)
        child_node = MCTSNode(new_board, self.network, parent=self, move=move)
        self.children[move] = child_node
        return child_node

    def backup(self, value):
        """Backpropagates the evaluation value up the tree."""
        self.visits += 1
        self.value_sum += value
        if self.parent:
            # Invert the value for the opponent.
            self.parent.backup(-value)

def mcts_search(board, network, action_space, num_simulations=100):
    root = MCTSNode(board, network, action_space)
    for _ in range(num_simulations):
        node = root
        while not node.is_terminal and node.is_fully_expanded():
            node = node.best_child()
        if not node.is_terminal:
            node = node.expand()
        if node.is_terminal:
            outcome = node.board.result()
            if outcome == "1-0":
                value = 1
            elif outcome == "0-1":
                value = -1
            else:
                value = 0
        else:
            board_tensor = network.board_to_tensor(node.board)
            _, value_tensor = network.forward(board_tensor)
            value = value_tensor.item()
        node.backup(value)
    
    best_move = max(root.children.items(), key=lambda item: item[1].visits)[0]
    return best_move
