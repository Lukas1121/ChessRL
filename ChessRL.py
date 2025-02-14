# chessRL.py
import random
import chess
import numpy as np
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use the first available GPU
    print("GPU is available and being used.")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU instead.")
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from minmax import get_best_move

class ChessRL:
    pawn_table = torch.tensor([
    [   0,    0,    0,    0,    0,    0,    0,    0],
    [  50,   50,   50,   50,   50,   50,   50,   50],
    [  10,   10,   20,   30,   30,   20,   10,   10],
    [   5,    5,   10,   25,   25,   10,    5,    5],
    [   0,    0,    0,   20,   20,    0,    0,    0],
    [   5,   -5,  -10,    0,    0,  -10,   -5,    5],
    [   5,   10,   10,  -20,  -20,   10,   10,    5],
    [   0,    0,    0,    0,    0,    0,    0,    0]
    ], dtype=torch.float) / 100.0

    # Knight piece-square table.
    knight_table = torch.tensor([
        [-50, -40, -30, -30, -30, -30, -40, -50],
        [-40, -20,   0,   0,   0,   0, -20, -40],
        [-30,   0,  10,  15,  15,  10,   0, -30],
        [-30,   5,  15,  20,  20,  15,   5, -30],
        [-30,   0,  15,  20,  20,  15,   0, -30],
        [-30,   5,  10,  15,  15,  10,   5, -30],
        [-40, -20,   0,   5,   5,   0, -20, -40],
        [-50, -40, -30, -30, -30, -30, -40, -50]
    ], dtype=torch.float) / 100.0

    # Bishop piece-square table.
    bishop_table = torch.tensor([
        [-20, -10, -10, -10, -10, -10, -10, -20],
        [-10,   0,   0,   0,   0,   0,   0, -10],
        [-10,   0,   5,  10,  10,   5,   0, -10],
        [-10,   5,   5,  10,  10,   5,   5, -10],
        [-10,   0,  10,  10,  10,  10,   0, -10],
        [-10,  10,  10,  10,  10,  10,  10, -10],
        [-10,   5,   0,   0,   0,   0,   5, -10],
        [-20, -10, -10, -10, -10, -10, -10, -20]
    ], dtype=torch.float) / 100.0

    # Rook piece-square table.
    rook_table = torch.tensor([
        [  0,   0,   0,   0,   0,   0,   0,   0],
        [  5,  10,  10,  10,  10,  10,  10,   5],
        [ -5,   0,   0,   0,   0,   0,   0,  -5],
        [ -5,   0,   0,   0,   0,   0,   0,  -5],
        [ -5,   0,   0,   0,   0,   0,   0,  -5],
        [ -5,   0,   0,   0,   0,   0,   0,  -5],
        [ -5,   0,   0,   0,   0,   0,   0,  -5],
        [  0,   0,   0,   5,   5,   0,   0,   0]
    ], dtype=torch.float) / 100.0

    # Queen piece-square table.
    queen_table = torch.tensor([
        [-20, -10, -10,  -5,  -5, -10, -10, -20],
        [-10,   0,   0,   0,   0,   0,   0, -10],
        [-10,   0,   5,   5,   5,   5,   0, -10],
        [ -5,   0,   5,   5,   5,   5,   0,  -5],
        [  0,   0,   5,   5,   5,   5,   0,  -5],
        [-10,   5,   5,   5,   5,   5,   0, -10],
        [-10,   0,   5,   0,   0,   0,   0, -10],
        [-20, -10, -10,  -5,  -5, -10, -10, -20]
    ], dtype=torch.float) / 100.0

    # King piece-square table (for the middlegame).
    king_table = torch.tensor([
        [-30, -40, -40, -50, -50, -40, -40, -30],
        [-30, -40, -40, -50, -50, -40, -40, -30],
        [-30, -40, -40, -50, -50, -40, -40, -30],
        [-30, -40, -40, -50, -50, -40, -40, -30],
        [-20, -30, -30, -40, -40, -30, -30, -20],
        [-10, -20, -20, -20, -20, -20, -20, -10],
        [ 20,  20,   0,   0,   0,   0,  20,  20],
        [ 20,  30,  10,   0,   0,  10,  30,  20]
    ], dtype=torch.float) / 10
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 20,
    }
    action_space = [chess.Move(from_sq, to_sq) for from_sq in chess.SQUARES for to_sq in chess.SQUARES]

    def __init__(self, board, color):
        self.board = board
        self.board_tensor = self.board_to_tensor(board)
        self.color = color

    def board_to_tensor(self, board):
        tensor = torch.zeros(12, 8, 8, dtype=torch.float)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                if piece.color == chess.WHITE:
                    channel = piece.piece_type - 1
                else:
                    channel = piece.piece_type - 1 + 6
                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)
                tensor[channel, row, col] = 1
        return tensor

    def create_legal_mask(self):
        legal_moves_set = {move.uci() for move in self.board.legal_moves}
        mask = torch.zeros(len(self.action_space), dtype=torch.float)
        for idx, move in enumerate(self.action_space):
            if move.uci() in legal_moves_set:
                mask[idx] = 1.0
        return mask
    
    def compute_material_score(self, board=None):
        """
        Evaluate the board from the perspective of self.color.
        
        This function sums up:
        - The material value of each piece (from self.PIECE_VALUES), and
        - The positional bonus/penalty from the corresponding piece-square table.
        
        For Black pieces, the piece-square table is flipped vertically since the tables are
        defined from White's perspective.
        
        Args:
            board (chess.Board): The board to evaluate. If None, uses self.board.
            
        Returns:
            float: The evaluation score. A positive score favors White and a negative score favors Black.
                The returned score is from the perspective of self.color.
        """
        if board is None:
            board = self.board

        total_value = 0.0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue

            # Get the base material value for the piece.
            material_value = self.PIECE_VALUES.get(piece.piece_type, 0)
            # Get the board coordinates (row and column).
            row = chess.square_rank(square)
            col = chess.square_file(square)

            # Determine positional bonus based on piece type and color.
            if piece.color == chess.WHITE:
                if piece.piece_type == chess.PAWN:
                    pos_value = self.pawn_table[row, col]
                elif piece.piece_type == chess.KNIGHT:
                    pos_value = self.knight_table[row, col]
                elif piece.piece_type == chess.BISHOP:
                    pos_value = self.bishop_table[row, col]
                elif piece.piece_type == chess.ROOK:
                    pos_value = self.rook_table[row, col]
                elif piece.piece_type == chess.QUEEN:
                    pos_value = self.queen_table[row, col]
                elif piece.piece_type == chess.KING:
                    pos_value = self.king_table[row, col]
                # Add value for White pieces.
                total_value += (material_value + pos_value)
            else:
                # For Black, flip the row since the tables are from White's perspective.
                flipped_row = 7 - row
                if piece.piece_type == chess.PAWN:
                    pos_value = self.pawn_table[flipped_row, col]
                elif piece.piece_type == chess.KNIGHT:
                    pos_value = self.knight_table[flipped_row, col]
                elif piece.piece_type == chess.BISHOP:
                    pos_value = self.bishop_table[flipped_row, col]
                elif piece.piece_type == chess.ROOK:
                    pos_value = self.rook_table[flipped_row, col]
                elif piece.piece_type == chess.QUEEN:
                    pos_value = self.queen_table[flipped_row, col]
                elif piece.piece_type == chess.KING:
                    pos_value = self.king_table[flipped_row, col]
                # Subtract value for Black pieces (since material is defined positively for White).
                total_value -= (material_value + pos_value)

        # Return the score from the perspective of self.color.
        # If self.color is Black, we invert the score so that a positive score always means "good" for the agent.
        return total_value if self.color == chess.WHITE else -total_value

    def update_board(self, board):
        self.board = board
        self.board_tensor = self.board_to_tensor(board)
        current_score = self.compute_material_score(board)
        # Compute the material difference from the previous state.
        self.material_delta = current_score - self.last_material_score
        # Update the last_material_score for the next comparison.
        self.last_material_score = current_score

class ChessPolicyNet(nn.Module, ChessRL):
    def __init__(self, num_actions, board, color, non_capture_penalty=-0.2, epsilon=0.1):
        nn.Module.__init__(self)
        ChessRL.__init__(self, board, color)
        # Define convolutional layers:
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Define fully connected layers:
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_actions)

        self.non_capture_penalty = non_capture_penalty
        self.epsilon = epsilon  # Exploration parameter.

        self.move_history = []
        self.last_material_score = self.compute_material_score(board)

    def forward(self):
        # Use the stored board tensor; assume it’s already updated.
        # Add batch dimension if necessary.
        x = self.board_tensor.to(device).unsqueeze(0)  # Now x has shape (1, 12, 8, 8)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        # Generate legal mask and apply it.
        legal_mask = self.create_legal_mask().to(device)  # Shape: (num_actions,)
        legal_mask = legal_mask.unsqueeze(0).expand_as(logits)
        masked_logits = logits + (legal_mask - 1) * 1e8
        probs = F.softmax(masked_logits, dim=-1)
        return probs

    def choose_move(self, use_minimax=False, minimax_depth=3):
        """
        Choose a move using either the RL network (with exploration) or minimax search.
        
        Args:
            use_minimax (bool): If True, use the minimax algorithm to select the move.
            minimax_depth (int): Depth to search if using minimax.
        
        Returns:
            chess.Move: The selected move.
        """
        if use_minimax:
            # Use minimax search with a depth limit. We pass a lambda wrapper for our evaluation function.
            move = get_best_move(self.board, minimax_depth, lambda board: self.compute_material_score(board))
            
            # Log the minimax move. Since minimax is not differentiable, we log a dummy log_prob.
            dummy_log_prob = torch.tensor(0.0, device=self.board_tensor.device)
            points = self.compute_material_score()
            self.move_history.append({
                'state_tensor': self.board_tensor,
                'probs': None,
                'action_index': None,
                'move': move,
                'log_prob': dummy_log_prob,
                'points': points,
                'minimax': True  # Mark that this move was chosen via minimax.
            })
            return move

        # Otherwise, proceed with the RL-based move selection.
        if np.random.rand() < self.epsilon:
            # Obtain legal moves from the board.
            legal_moves = list(self.board.legal_moves)
            # Randomly choose one.
            move = random.choice(legal_moves)

            # Log that this was a random move.
            dummy_log_prob = torch.tensor(0.0, device=self.board_tensor.device)
            points = self.compute_material_score()
            self.move_history.append({
                'state_tensor': self.board_tensor,
                'probs': None,  # Not used for random moves.
                'action_index': None,
                'move': move,
                'log_prob': dummy_log_prob,
                'points': points,
                'random': True  # Mark that this was chosen randomly.
            })
            return move
        else:
            # Use the policy network to select a move.
            probs = self.forward()  # Expected shape: (1, num_actions)
            m = D.Categorical(probs)
            action = m.sample()      # Keep this as a tensor.
            action_index = action.item()
            move = self.action_space[action_index]

            # Compute the log probability that remains attached to the computation graph.
            log_prob = m.log_prob(action).to(self.board_tensor.device)
            points = self.compute_material_score()

            self.move_history.append({
                'state_tensor': self.board_tensor,
                'probs': probs,
                'action_index': action_index,
                'move': move,
                'log_prob': log_prob,
                'points': points,
                'random': False  # Mark that this was chosen by the network.
            })

            return move


def reinforce_update(policy_net, optimizer, gamma=0.99):
    """
    Perform a REINFORCE update using the move history stored in policy_net.move_history.

    Each move entry should contain:
      - 'log_prob': A tensor containing the log probability (requires grad).
      - 'points': The reward for that move.

    Args:
        policy_net: The policy network instance (e.g., for White or Black).
        optimizer: The optimizer to update policy_net parameters.
        gamma (float): Discount factor.

    Returns:
        float: The loss value.
    """
    # 1. Extract rewards from the move history.
    rewards = [entry['points'] for entry in policy_net.move_history]

    # 2. Compute cumulative discounted rewards (returns).
    returns_list = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns_list.insert(0, R)

    # Create the returns tensor on the same device as the model.
    returns = torch.tensor(returns_list, dtype=torch.float32, device=policy_net.board_tensor.device)
    # Squeeze the returns tensor so that each element is a scalar.
    returns = returns.squeeze()

    # Optionally normalize returns.
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)

    # 3. Compute the REINFORCE loss.
    loss = 0
    for i, entry in enumerate(policy_net.move_history):
        # Squeeze the log probability so that it is a scalar.
        log_prob = entry['log_prob'].squeeze()
        loss += -log_prob * returns[i].squeeze()  # Ensure both are scalars.

    # 4. Backpropagate and update the network.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_val = loss.item()
    print(f"REINFORCE loss: {loss_val:.4f}")

    # Clear the move history for the next batch.
    policy_net.move_history = []
    return loss_val
