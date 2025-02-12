# chessRL.py
import os
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
    ], dtype=torch.float) / 200.0
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 200,
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


    # def compute_material_score(self, board=None):
    #     """
    #     Compute the material difference from the perspective of self.color.

    #     If self.color == chess.WHITE, returns (White points - Black points).
    #     If self.color == chess.BLACK, returns (Black points - White points).

    #     Args:
    #         board (chess.Board): Optional board to evaluate. If None, uses self.board.

    #     Returns:
    #         int: The material difference for the specified color.
    #     """
    #     if board is None:
    #         board = self.board

    #     white_score = 0
    #     black_score = 0
    #     for square in chess.SQUARES:
    #         piece = board.piece_at(square)
    #         if piece is not None:
    #             value = self.PIECE_VALUES.get(piece.piece_type, 0)
    #             if piece.color == chess.WHITE:
    #                 white_score += value
    #             else:
    #                 black_score += value

    #     if self.color == chess.WHITE:
    #         return white_score - black_score
    #     else:  # self.color == chess.BLACK
    #         return black_score - white_score

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

    def choose_move(self):
        # With probability epsilon, choose a random legal move.
        if np.random.rand() < self.epsilon:
            # Obtain legal moves from the board.
            legal_moves = list(self.board.legal_moves)
            # Randomly choose one.
            move = random.choice(legal_moves)

            # Optionally, log that this was a random move.
            # Here, we log a dummy log probability (or you might mark it in the history).
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
            # Otherwise, use the policy network to select a move.
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
    ], dtype=torch.float) / 200.0
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 200,
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


    # def compute_material_score(self, board=None):
    #     """
    #     Compute the material difference from the perspective of self.color.

    #     If self.color == chess.WHITE, returns (White points - Black points).
    #     If self.color == chess.BLACK, returns (Black points - White points).

    #     Args:
    #         board (chess.Board): Optional board to evaluate. If None, uses self.board.

    #     Returns:
    #         int: The material difference for the specified color.
    #     """
    #     if board is None:
    #         board = self.board

    #     white_score = 0
    #     black_score = 0
    #     for square in chess.SQUARES:
    #         piece = board.piece_at(square)
    #         if piece is not None:
    #             value = self.PIECE_VALUES.get(piece.piece_type, 0)
    #             if piece.color == chess.WHITE:
    #                 white_score += value
    #             else:
    #                 black_score += value

    #     if self.color == chess.WHITE:
    #         return white_score - black_score
    #     else:  # self.color == chess.BLACK
    #         return black_score - white_score

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

    def choose_move(self):
        # With probability epsilon, choose a random legal move.
        if np.random.rand() < self.epsilon:
            # Obtain legal moves from the board.
            legal_moves = list(self.board.legal_moves)
            # Randomly choose one.
            move = random.choice(legal_moves)

            # Optionally, log that this was a random move.
            # Here, we log a dummy log probability (or you might mark it in the history).
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
            # Otherwise, use the policy network to select a move.
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

def play_self_game(
    policy_net_white, 
    policy_net_black, 
    terminal_reward=100, 
    terminal_loss=100,
    per_move_penalty=0,
    move_length_threshold=200,         # Maximum number of moves allowed.
    exceeding_length_penalty=100        # Extra penalty if the game exceeds the threshold.
):
    """
    Play a self-play game between two policy networks, forcefully ending the game
    if it exceeds move_length_threshold, and adjust terminal rewards accordingly.
    
    Returns:
        tuple: (white_move_history, black_move_history, game_length, result)
    """
    # Initialize the board.
    board = chess.Board()
    
    # Play the game.
    while not board.is_game_over():
        # Forcefully end the game if move threshold exceeded.
        if len(policy_net_white.move_history) >= move_length_threshold:
            print("Move threshold exceeded. Forcing game end.")
            break
        
        if board.turn == chess.WHITE:
            policy_net_white.update_board(board)
            previous_material = policy_net_white.compute_material_score()
            move = policy_net_white.choose_move()
            if move not in board.legal_moves:
                continue
            board.push(move)
            policy_net_white.update_board(board)
            current_material = policy_net_white.compute_material_score()
            material_change = current_material - previous_material
            if material_change == 0:
                policy_net_white.move_history[-1]['points'] += policy_net_white.non_capture_penalty
        else:
            policy_net_black.update_board(board)
            previous_material = policy_net_black.compute_material_score()
            move = policy_net_black.choose_move()
            if move not in board.legal_moves:
                continue
            board.push(move)
            policy_net_black.update_board(board)
            current_material = policy_net_black.compute_material_score()
            material_change = current_material - previous_material
            if material_change == 0:
                policy_net_black.move_history[-1]['points'] += policy_net_black.non_capture_penalty

    # Optionally, print the final board.
    # print_custom_board(board)
    print("Game over!")
    result = board.result()
    print("Result:", result)
    
    # Compute the game length.
    game_length = len(policy_net_white.move_history)
    
    # Compute the total penalty based on the game length.
    length_penalty = per_move_penalty * game_length

    # Adjust the terminal rewards based on the game result.
    if result == "1-0":
        terminal_reward_white = terminal_reward
        terminal_reward_black = -terminal_loss
    elif result == "0-1":
        terminal_reward_white = -terminal_loss
        terminal_reward_black = terminal_reward
    else:
        terminal_reward_white = -length_penalty
        terminal_reward_black = -length_penalty

    # If the game exceeded the threshold, apply an extra penalty.
    if game_length >= move_length_threshold:
        terminal_reward_white -= exceeding_length_penalty
        terminal_reward_black -= exceeding_length_penalty

    # Apply these terminal rewards to the last move in each move history.
    if policy_net_white.move_history:
        policy_net_white.move_history[-1]['points'] = policy_net_white.move_history[-1].get('points', 0) + terminal_reward_white
    if policy_net_black.move_history:
        policy_net_black.move_history[-1]['points'] = policy_net_black.move_history[-1].get('points', 0) + terminal_reward_black

    return policy_net_white.move_history, policy_net_black.move_history, game_length, result

import datetime

def train_chess_policy_networks(
    num_iterations=400,
    games_per_iteration=5,
    epsilon_initial=0.3,
    epsilon_final=0.1,
    lr=0.0004,
    gamma=0.95,
    terminal_reward=1000,
    terminal_loss=100,
    per_move_penalty=1,
    non_capture_penalty=-1,
    move_length_threshold=200,         # New parameter: move threshold
    exceeding_length_penalty=1000,       # New parameter: penalty if game exceeds move_length_threshold
    pretrained_model_path_white=None,
    pretrained_model_path_black=None,
    save_path_white="policy_net_white.pth",
    save_path_black="policy_net_black.pth",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    Train chess policy networks using self-play and the REINFORCE algorithm.

    Parameters:
        num_iterations (int): Number of training iterations (batches).
        games_per_iteration (int): Number of self-play games per iteration.
        epsilon_initial (float): Starting exploration probability.
        epsilon_final (float): Minimum exploration probability.
        lr (float): Learning rate for the optimizers.
        gamma (float): Discount factor.
        terminal_reward (float): Reward given at the terminal state.
        terminal_loss (float): Reward (or loss) given at the terminal state on losing.
        per_move_penalty (float): Penalty per move taken.
        non_capture_penalty (float): Penalty for non-capturing moves.
        move_length_threshold (int): Maximum number of moves allowed before applying a huge penalty.
        exceeding_length_penalty (float): The extra penalty applied if the game exceeds move_length_threshold.
        pretrained_model_path_white (str or None): Path to the pretrained White model.
        pretrained_model_path_black (str or None): Path to the pretrained Black model.
        save_path_white (str): Path to save the trained White policy network.
        save_path_black (str): Path to save the trained Black policy network.
        device (torch.device): Device on which to run the training.

    Returns:
        dict: A dictionary containing the trained models and training metrics.
    """
    # Instantiate the policy networks for White and Black.
    policy_net_white = ChessPolicyNet(
        num_actions=len(ChessRL.action_space),
        board=chess.Board(),
        color=chess.WHITE,
        non_capture_penalty=non_capture_penalty,
        epsilon=epsilon_initial
    ).to(device)

    policy_net_black = ChessPolicyNet(
        num_actions=len(ChessRL.action_space),
        board=chess.Board(),
        color=chess.BLACK,
        non_capture_penalty=non_capture_penalty,
        epsilon=epsilon_initial
    ).to(device)

    # Load the pretrained models into the networks if provided.
    if pretrained_model_path_white is not None:
        white_state_dict = torch.load(pretrained_model_path_white, map_location=device)
        policy_net_white.load_state_dict(white_state_dict)
    else:
        print("No pretrained White model provided. Training from scratch.")

    if pretrained_model_path_black is not None:
        black_state_dict = torch.load(pretrained_model_path_black, map_location=device)
        policy_net_black.load_state_dict(black_state_dict)
    else:
        print("No pretrained Black model provided. Training from scratch.")

    policy_net_white.train()
    policy_net_black.train()

    # Set up optimizers.
    optimizer_white = torch.optim.Adam(policy_net_white.parameters(), lr=lr)
    optimizer_black = torch.optim.Adam(policy_net_black.parameters(), lr=lr)

    # Initialize lists to store metrics.
    white_avg_points = []
    black_avg_points = []
    white_loss_list = []
    black_loss_list = []
    white_avg_entropy = []
    black_avg_entropy = []
    white_avg_log_probs = []
    black_avg_log_probs = []
    game_length_list = []
    white_win_rates = []
    black_win_rates = []
    draw_rates = []

    # Main training loop.
    for iteration in range(num_iterations):
        # Linearly decay epsilon.
        epsilon = max(epsilon_final, epsilon_initial - (epsilon_initial - epsilon_final) * (iteration / num_iterations))
        policy_net_white.epsilon = epsilon
        policy_net_black.epsilon = epsilon

        print(f"Starting iteration {iteration+1}/{num_iterations} with epsilon = {epsilon:.4f}")

        # Batch data for this iteration.
        batch_white_move_history = []
        batch_black_move_history = []
        game_lengths = []
        results = []  # Expected results: "1-0", "0-1", or "1/2-1/2".

        # Play self-play games.
        for _ in range(games_per_iteration):
            # play_self_game now receives extra parameters for move length penalty.
            white_move_history, black_move_history, game_length, result = play_self_game(
                policy_net_white, policy_net_black,
                per_move_penalty=per_move_penalty,
                terminal_reward=terminal_reward,
                terminal_loss=terminal_loss,
                move_length_threshold=move_length_threshold,
                exceeding_length_penalty=exceeding_length_penalty
            )
            batch_white_move_history.extend(white_move_history)
            batch_black_move_history.extend(black_move_history)
            game_lengths.append(game_length)
            results.append(result)

        # Assign the collected move histories to each network.
        policy_net_white.move_history = batch_white_move_history
        policy_net_black.move_history = batch_black_move_history

        # Perform the REINFORCE update.
        white_loss = reinforce_update(policy_net_white, optimizer_white, gamma=gamma)
        black_loss = reinforce_update(policy_net_black, optimizer_black, gamma=gamma)
        white_loss_list.append(white_loss)
        black_loss_list.append(black_loss)

        # Calculate average points per move.
        white_avg = np.mean([entry['points'] for entry in batch_white_move_history])
        black_avg = np.mean([entry['points'] for entry in batch_black_move_history])
        white_avg_points.append(white_avg)
        black_avg_points.append(black_avg)

        # Compute average entropy and average log probability for chosen moves.
        white_entropies = [compute_entropy(entry['probs']) for entry in batch_white_move_history if entry['probs'] is not None]
        black_entropies = [compute_entropy(entry['probs']) for entry in batch_black_move_history if entry['probs'] is not None]
        white_avg_entropy.append(np.mean(white_entropies) if white_entropies else 0)
        black_avg_entropy.append(np.mean(black_entropies) if black_entropies else 0)

        white_log_probs = [entry['log_prob'].item() for entry in batch_white_move_history if entry['log_prob'] is not None]
        black_log_probs = [entry['log_prob'].item() for entry in batch_black_move_history if entry['log_prob'] is not None]
        white_avg_log_probs.append(np.mean(white_log_probs) if white_log_probs else 0)
        black_avg_log_probs.append(np.mean(black_log_probs) if black_log_probs else 0)

        # Average game length.
        avg_game_length = np.mean(game_lengths)
        game_length_list.append(avg_game_length)

        # Compute win, loss, and draw rates.
        white_wins = sum(1 for r in results if r == "1-0")
        draws = sum(1 for r in results if r == "1/2-1/2")
        white_losses = sum(1 for r in results if r == "0-1")
        total_games = len(results)
        win_rate_white = white_wins / total_games
        draw_rate = draws / total_games
        win_rate_black = white_losses / total_games  # Black wins when white loses.
        white_win_rates.append(win_rate_white)
        black_win_rates.append(win_rate_black)
        draw_rates.append(draw_rate)

        print(f"Iteration {iteration+1} complete.")
        print(f"  Avg Game Length: {avg_game_length:.2f}")
        print(f"  White Avg Points: {white_avg:.2f}, Black Avg Points: {black_avg:.2f}")
        print(f"  White Win Rate: {win_rate_white:.2f}, Black Win Rate: {win_rate_black:.2f}, Draw Rate: {draw_rate:.2f}\n")

    # -------------------- Save models and parameters --------------------
    # Create a folder with a timestamp to store the .pth files and parameters.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = os.path.join(os.path.dirname(save_path_white), f"run_{timestamp}")
    os.makedirs(save_folder, exist_ok=True)

    # Create new file paths within the folder.
    white_save_path = os.path.join(save_folder, "policy_net_white.pth")
    black_save_path = os.path.join(save_folder, "policy_net_black.pth")

    # Save the trained models.
    torch.save(policy_net_white.state_dict(), white_save_path)
    torch.save(policy_net_black.state_dict(), black_save_path)
    print("Models saved in folder:", save_folder)

    # Save the input parameters to a text file.
    params = {
        "num_iterations": num_iterations,
        "games_per_iteration": games_per_iteration,
        "epsilon_initial": epsilon_initial,
        "epsilon_final": epsilon_final,
        "lr": lr,
        "gamma": gamma,
        "terminal_reward": terminal_reward,
        "terminal_loss": terminal_loss,
        "per_move_penalty": per_move_penalty,
        "non_capture_penalty": non_capture_penalty,
        "move_length_threshold": move_length_threshold,
        "exceeding_length_penalty": exceeding_length_penalty,
        "pretrained_model_path_white": pretrained_model_path_white,
        "pretrained_model_path_black": pretrained_model_path_black,
        "save_path_white": save_path_white,
        "save_path_black": save_path_black,
        "device": str(device)
    }

    params_file = os.path.join(save_folder, "parameters.txt")
    with open(params_file, "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    print("Parameters saved to:", params_file)
    # ---------------------------------------------------------------------

    # Package metrics and models in a dictionary for return.
    metrics = {
        "policy_net_white": policy_net_white,
        "policy_net_black": policy_net_black,
        "white_loss_list": white_loss_list,
        "black_loss_list": black_loss_list,
        "white_avg_points": white_avg_points,
        "black_avg_points": black_avg_points,
        "white_avg_entropy": white_avg_entropy,
        "black_avg_entropy": black_avg_entropy,
        "white_avg_log_probs": white_avg_log_probs,
        "black_avg_log_probs": black_avg_log_probs,
        "game_length_list": game_length_list,
        "white_win_rates": white_win_rates,
        "black_win_rates": black_win_rates,
        "draw_rates": draw_rates
    }

    return metrics