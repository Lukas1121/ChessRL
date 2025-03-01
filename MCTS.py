import numpy as np
import random
import chess
import torch  # Assuming torch is imported and available

class MCTSNode:
    def __init__(self, board, network, action_space, parent=None, move=None):
        self.board = board.copy()
        self.network = network
        self.action_space = action_space  # List of all possible moves.
        self.parent = parent
        self.move = move  # The move that led to this node.
        self.children = {}  # Maps moves to child nodes.
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
        """Selects the best child using a PUCT formula that balances exploitation and exploration."""
        best_score = -float('inf')
        best_child = None
        for move, child in self.children.items():
            # Q: average value; U: exploration bonus.
            Q = child.value_sum / (child.visits + 1e-8)
            U = c_puct * self.priors[move] * np.sqrt(self.visits) / (1 + child.visits)
            score = Q + U
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self):
        """Expands one child that has not been added to the tree yet."""
        legal_moves = list(self.board.legal_moves)
        unexpanded_moves = [move for move in legal_moves if move not in self.children]
        if not unexpanded_moves:
            return None
        move = random.choice(unexpanded_moves)
        new_board = self.board.copy()
        new_board.push(move)
        child_node = MCTSNode(new_board, self.network, self.action_space, parent=self, move=move)
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
    """
    Runs MCTS for a given number of simulations using batched network evaluations for nonterminal leaf nodes.
    Prints a single summary message upon completion.
    """
    root = MCTSNode(board, network, action_space)
    leaf_nodes = []  # Will store nodes that need to be batch-evaluated.

    for sim in range(num_simulations):
        node = root
        # Selection: traverse the tree until reaching a nonterminal node that isn't fully expanded.
        while not node.is_terminal and node.is_fully_expanded():
            node = node.best_child()
        # Terminal node: evaluate immediately.
        if node.is_terminal:
            outcome = node.board.result()
            if outcome == "1-0":
                value = 1
            elif outcome == "0-1":
                value = -1
            else:
                value = 0
            node.backup(value)
        else:
            # Expansion: expand one child and add it to the batch queue.
            expanded_node = node.expand()
            leaf_nodes.append(expanded_node)

    # Batch evaluate all collected leaf nodes.
    if leaf_nodes:
        board_tensors = torch.stack([network.board_to_tensor(node.board) for node in leaf_nodes])
        # Remove any extra dimensions until the tensor is 4D.
        while board_tensors.ndim > 4:
            board_tensors = board_tensors.squeeze(0)
        _, value_tensors = network.forward(board_tensors)
        for node, value_tensor in zip(leaf_nodes, value_tensors):
            node.backup(value_tensor.item())

    # Compute policy information from the root's children using normalized visit counts.
    total_visits = sum(child.visits for child in root.children.values())
    policy_info = np.zeros(len(action_space))
    for move, child in root.children.items():
        index = action_space.index(move)
        policy_info[index] = child.visits / total_visits if total_visits > 0 else 0

    best_move = max(root.children.items(), key=lambda item: item[1].visits)[0]
    # Print one summary message to help track progress for the current game.
    # print(f"MCTS complete: {num_simulations} simulations run, {len(leaf_nodes)} leaf nodes evaluated. Best move: {best_move}")
    return best_move, policy_info
