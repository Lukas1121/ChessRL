import chess
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from MCTS import mcts_search

def create_action_space():
    """
    Create a complete action space for chess moves including:
    - Basic moves (all from-square to to-square combinations)
    - Pawn promotion moves (for both White and Black)
    - Castling moves (which are generated in the basic moves)

    Returns:
        list[chess.Move]: A list of chess.Move objects representing the full action space.
    """
    action_space = []

    # 1. Basic moves: Iterate over all possible from and to squares.
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            move = chess.Move(from_sq, to_sq)
            if move not in action_space:
                action_space.append(move)

    # 2. Add promotion moves for White:
    # White pawn promotions occur when a pawn moves from rank 7 (index 6) to rank 8 (index 7).
    for from_sq in chess.SQUARES:
        if chess.square_rank(from_sq) == 6:  # White pawn on 7th rank (0-indexed)
            for to_sq in chess.SQUARES:
                if chess.square_rank(to_sq) == 7:  # Destination is 8th rank
                    # Typically a pawn moves straight ahead or diagonally.
                    if abs(chess.square_file(from_sq) - chess.square_file(to_sq)) <= 1:
                        for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                            promo_move = chess.Move(from_sq, to_sq, promotion=promo)
                            if promo_move not in action_space:
                                action_space.append(promo_move)

    # 3. Add promotion moves for Black:
    # Black pawn promotions occur when a pawn moves from rank 2 (index 1) to rank 1 (index 0).
    for from_sq in chess.SQUARES:
        if chess.square_rank(from_sq) == 1:  # Black pawn on 2nd rank
            for to_sq in chess.SQUARES:
                if chess.square_rank(to_sq) == 0:  # Destination is 1st rank
                    if abs(chess.square_file(from_sq) - chess.square_file(to_sq)) <= 1:
                        for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                            promo_move = chess.Move(from_sq, to_sq, promotion=promo)
                            if promo_move not in action_space:
                                action_space.append(promo_move)

    return action_space

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
    ], dtype=torch.float) / 100
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }
    action_space = create_action_space()
    move_to_idx = {move.uci(): idx for idx, move in enumerate(action_space)}

    # Precompute a mapping from square to (row, col)
    SQUARE_TO_RC = {square: (7 - chess.square_rank(square), chess.square_file(square))
                    for square in chess.SQUARES}

    def __init__(self, board, color):
        self.board = board
        self.board_tensor = self.board_to_tensor(board)
        self.value_tensor = self.precompute_value_tensor() 
        self.color = color

    def board_to_tensor(self, board):
        tensor = torch.zeros(12, 8, 8, dtype=torch.float)
        for square, piece in board.piece_map().items():
            row, col = self.SQUARE_TO_RC[square]
            channel = piece.piece_type - 1 + (0 if piece.color == chess.WHITE else 6)
            tensor[channel, row, col] = 1
        return tensor
    
    def precompute_value_tensor(self):
        # Create a tensor of shape [12, 8, 8] for piece values (material + positional)
        value_tensor = torch.zeros((12, 8, 8), dtype=torch.float)
        for channel in range(12):
            piece_type = (channel % 6) + 1  # Chess pieces are 1-indexed
            is_white = channel < 6

            material_val = self.PIECE_VALUES[piece_type]
            # Select the appropriate piece-square table based on piece type.
            if piece_type == chess.PAWN:
                table = self.pawn_table
            elif piece_type == chess.KNIGHT:
                table = self.knight_table
            elif piece_type == chess.BISHOP:
                table = self.bishop_table
            elif piece_type == chess.ROOK:
                table = self.rook_table
            elif piece_type == chess.QUEEN:
                table = self.queen_table
            elif piece_type == chess.KING:
                table = self.king_table

            if is_white:
                # For white, use the table as-is.
                value_tensor[channel] = material_val + table
            else:
                # For black, flip the table vertically and negate the value.
                value_tensor[channel] = -(material_val + torch.flip(table, dims=[0]))
        return value_tensor
    
    def compute_material_score(self):
        # Compute the score using vectorized multiplication.
        total_score = torch.sum(self.board_tensor * self.value_tensor)
        # Return the score from the perspective of the agent's color.
        return total_score if self.color == chess.WHITE else -total_score       

    def create_legal_mask(self):
        # Create a mask of zeros
        mask = torch.zeros(len(self.action_space), dtype=torch.float)
        # Loop only over legal moves
        for move in self.board.legal_moves:
            uci_str = move.uci()
            if uci_str in self.move_to_idx:
                idx = self.move_to_idx[uci_str]
                mask[idx] = 1.0
        return mask

    def update_board(self, board):
        self.board = board
        self.board_tensor = self.board_to_tensor(board)

class ChessPositionalEvaluator(nn.Module):
    def __init__(self, device):
        super(ChessPositionalEvaluator, self).__init__()

        self.device = device
        
        # Initialize opening, middlegame, and endgame tables
        self.phases = ['opening', 'middlegame', 'endgame']
        
        self.pawn_table_opening = nn.Parameter(self._init_table(ChessRL.pawn_table, emphasis='center'))
        self.knight_table_opening = nn.Parameter(self._init_table(ChessRL.knight_table, emphasis='development'))
        self.bishop_table_opening = nn.Parameter(self._init_table(ChessRL.bishop_table, emphasis='development'))
        self.rook_table_opening = nn.Parameter(self._init_table(ChessRL.rook_table, emphasis='development'))
        self.queen_table_opening = nn.Parameter(self._init_table(ChessRL.queen_table, emphasis='development'))
        self.king_table_opening = nn.Parameter(self._init_table(ChessRL.king_table, emphasis='kingside'))  # Slight preference for kingside castling

        # Middlegame phase tables (your existing tables are a good start)
        self.pawn_table_middlegame = nn.Parameter(torch.tensor(ChessRL.pawn_table, device=device))
        self.knight_table_middlegame = nn.Parameter(self._init_table(ChessRL.knight_table, emphasis='activity'))
        self.bishop_table_middlegame = nn.Parameter(self._init_table(ChessRL.bishop_table, emphasis='activity'))
        self.rook_table_middlegame = nn.Parameter(torch.tensor(ChessRL.rook_table, device=device))
        self.queen_table_middlegame = nn.Parameter(torch.tensor(ChessRL.queen_table, device=device))
        self.king_table_middlegame = nn.Parameter(torch.tensor(ChessRL.king_table, device=device))
        
        # Endgame phase tables (emphasize king centralization, pawn advancement)
        self.pawn_table_endgame = nn.Parameter(self._init_table(ChessRL.pawn_table, emphasis='advancement'))
        self.knight_table_endgame = nn.Parameter(self._init_table(ChessRL.knight_table, emphasis='activity'))
        self.bishop_table_endgame = nn.Parameter(self._init_table(ChessRL.bishop_table, emphasis='activity'))
        self.rook_table_endgame = nn.Parameter(self._init_table(ChessRL.rook_table, emphasis='activity'))
        self.queen_table_endgame = nn.Parameter(torch.tensor(ChessRL.queen_table, device=device))
        self.king_table_endgame = nn.Parameter(self._init_endgame_king_table())
        
        # Phase transition thresholds (learnable)
        self.opening_threshold = nn.Parameter(torch.tensor([30.0], device=device))  # Total material to end opening
        self.middlegame_threshold = nn.Parameter(torch.tensor([15.0], device=device))  # Total material to end middlegame
        
        # Material values (could also be learnable)
        self.piece_values = nn.Parameter(
            torch.tensor([1.0, 3.0, 3.0, 5.0, 9.0, 0.0], dtype=torch.float, device=device)
        )
    
    def _init_table(self, base_table, emphasis=None):
        """Initialize a table with optional emphasis on certain characteristics"""
        table = base_table.clone().to(self.device)
        
        if emphasis == 'center':
            # Boost center squares
            center_mask = torch.zeros_like(table)
            center_mask[3:5, 3:5] = 1.0
            table += center_mask * 0.05
            
        elif emphasis == 'development':
            # Boost development squares
            develop_mask = torch.zeros_like(table)
            develop_mask[0, 1:7] = 1.0  # Back rank except rook squares
            table += develop_mask * 0.1
            
        elif emphasis == 'kingside':
            # Boost kingside castling preparation
            kingside_mask = torch.zeros_like(table)
            kingside_mask[0, 5:7] = 1.0  # f1, g1 squares
            table += kingside_mask * 0.15
            
        elif emphasis == 'queenside':
            # Boost queenside castling preparation
            queenside_mask = torch.zeros_like(table)
            queenside_mask[0, 1:4] = 1.0  # b1, c1, d1 squares
            table += queenside_mask * 0.12
            
        elif emphasis == 'activity':
            # Boost piece activity (middle of board)
            activity_mask = torch.zeros_like(table)
            activity_mask[2:6, 2:6] = 1.0  # Middle 4x4 squares
            table += activity_mask * 0.08
            
        elif emphasis == 'advancement':
            # Boost advancement (forward ranks)
            for r in range(8):
                # Higher value for more advanced positions
                advancement_value = r * 0.01  # Small incremental bonus
                advancement_mask = torch.zeros_like(table)
                advancement_mask[r, :] = 1.0
                table += advancement_mask * advancement_value
                
        return table
    
    def _init_endgame_king_table(self):
        """Initialize an endgame king table that encourages centralization"""
        table = torch.zeros((8, 8), dtype=torch.float, device=self.device)
        for r in range(8):
            for c in range(8):
                # Distance from center (3.5, 3.5)
                center_dist = max(abs(r - 3.5), abs(c - 3.5))
                table[r, c] = (4 - center_dist) / 20.0  # Normalize values
        return table
    
    def _init_endgame_pawn_table(self):
        """Initialize an endgame pawn table that rewards advancement"""
        table = torch.zeros((8, 8), dtype=torch.float, device=self.device)
        for r in range(8):
            # Higher value for more advanced pawns
            advancement_value = r / 10.0
            table[r, :] = advancement_value
        return table
    
    def detect_game_phase(self, board_tensor):
        """Detect game phase based on material and move number"""
        # Count total material
        total_material = 0
        for piece_type in range(1, 6):  # Exclude king
            white_pieces = torch.sum(board_tensor[piece_type - 1])
            black_pieces = torch.sum(board_tensor[piece_type - 1 + 6])
            total_material += (white_pieces + black_pieces) * self.piece_values[piece_type - 1]
        
        # Calculate phase weights
        opening_weight = torch.sigmoid(self.opening_threshold - total_material)
        endgame_weight = torch.sigmoid(total_material - self.middlegame_threshold)
        middlegame_weight = 1.0 - opening_weight - endgame_weight
        
        return opening_weight, middlegame_weight, endgame_weight
    
    def get_blended_table(self, piece_type, phase_weights):
        """Get a phase-blended table for a specific piece type"""
        opening_w, mid_w, end_w = phase_weights
        
        if piece_type == chess.PAWN:
            return (opening_w * self.pawn_table_opening + 
                    mid_w * self.pawn_table_middlegame + 
                    end_w * self.pawn_table_endgame)
        elif piece_type == chess.KNIGHT:
            return (opening_w * self.knight_table_opening + 
                    mid_w * self.knight_table_middlegame + 
                    end_w * self.knight_table_endgame)
        elif piece_type == chess.BISHOP:
            return (opening_w * self.bishop_table_opening + 
                    mid_w * self.bishop_table_middlegame + 
                    end_w * self.bishop_table_endgame)
        elif piece_type == chess.ROOK:
            return (opening_w * self.rook_table_opening + 
                    mid_w * self.rook_table_middlegame + 
                    end_w * self.rook_table_endgame)
        elif piece_type == chess.QUEEN:
            return (opening_w * self.queen_table_opening + 
                    mid_w * self.queen_table_middlegame + 
                    end_w * self.queen_table_endgame)
        elif piece_type == chess.KING:
            return (opening_w * self.king_table_opening + 
                    mid_w * self.king_table_middlegame + 
                    end_w * self.king_table_endgame)
        else:
            # Default case to prevent None returns
            return torch.zeros((8, 8), device=self.device)
    
    def forward(self, board_tensor, color):
        """Compute positional value using phase-specific tables"""
        # Cache phase weights and blended tables for repeated calls
        if not hasattr(self, '_cached_phase') or self._cached_phase is None:
            self._cached_phase = {}
            self._cached_tables = {}
        
        # Create a cache key based on material count
        material_counts = []
        for piece_type in range(1, 6):  # Exclude king
            white_count = torch.sum(board_tensor[piece_type - 1]).item()
            black_count = torch.sum(board_tensor[piece_type - 1 + 6]).item()
            material_counts.append((piece_type, white_count, black_count))
        
        cache_key = tuple(material_counts)
        
        # Use cached phase weights and tables if available
        if cache_key in self._cached_phase:
            phase_weights = self._cached_phase[cache_key]
            blended_tables = self._cached_tables[cache_key]
        else:
            # Calculate phase weights
            phase_weights = self.detect_game_phase(board_tensor)
            
            # Precompute blended tables for all piece types
            blended_tables = {}
            for piece_type in range(1, 7):  # All pieces including king
                blended_tables[piece_type] = self.get_blended_table(piece_type, phase_weights)
            
            # Cache results
            self._cached_phase[cache_key] = phase_weights
            self._cached_tables[cache_key] = blended_tables
        
        # Compute the final value tensor
        value_tensor = torch.zeros_like(board_tensor)
        for channel in range(12):
            piece_type = (channel % 6) + 1
            is_white = channel < 6
            
            # Material value
            material_val = self.piece_values[piece_type - 1]
            
            # Get blended table for this piece type
            table = blended_tables[piece_type]
            
            if is_white:
                value_tensor[channel] = material_val + table
            else:
                value_tensor[channel] = -(material_val + torch.flip(table, dims=[0]))
        
        # Compute final value
        total_value = torch.sum(board_tensor * value_tensor)
        return total_value if color == chess.WHITE else -total_value

class ChessPolicyNet(nn.Module, ChessRL):
    def __init__(self, board, color, device, layers=5, epsilon=0.1, evaluator=None):
        nn.Module.__init__(self)
        ChessRL.__init__(self, board, color)
        self.device = device
        self.epsilon = epsilon

        # Create evaluator or use provided one
        self.evaluator = evaluator if evaluator is not None else ChessPositionalEvaluator(device)

        conv_layers = []
        # First layer: convert input channels (12) to 128
        conv_layers.append(nn.Conv2d(in_channels=12, out_channels=128, kernel_size=3, padding=1))
        conv_layers.append(nn.ReLU())
        
        if layers > 2:
            # Add (layers - 2) intermediate layers maintaining 128 channels
            for _ in range(layers - 2):
                conv_layers.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1))
                conv_layers.append(nn.ReLU())
        
        # Keep the last layer at 128 channels as well
        if layers > 1:
            conv_layers.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # After the conv layers, the spatial dimensions remain 8x8
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.relu = nn.ReLU()
        
        num_actions = len(ChessRL.action_space)
        self.policy_head = nn.Linear(512, num_actions)
        self.value_head = nn.Linear(512, 1)

    def forward(self,x=None):
        if x is None:
            x = self.board_tensor.to(self.device).unsqueeze(0)  # Add batch dimension
        
        # Get positional evaluation from the evaluator
        pos_value = self.evaluator(x.squeeze(0), self.color)
        
        # Process through convolutional layers
        x = self.conv_layers(x)
        
        # Flatten the conv output
        x_flat = x.view(x.size(0), -1)  # Shape: [batch_size, 128*8*8]
        
        # Process through fully connected layer
        x = self.relu(self.fc1(x_flat))
        
        # Output policy logits and value
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value, pos_value

    def choose_move(self):
        """Selects a move using epsilon-greedy RL policy."""
        # Get policy logits from the network
        policy_logits, value, pos_value = self.forward()
        
        # Apply epsilon-greedy strategy
        if np.random.rand() < self.epsilon:
            # Random move
            legal_moves = list(self.board.legal_moves)
            move = random.choice(legal_moves)
            
            # Create safe log probability
            # Find the index for this move in the action space
            try:
                action_index = self.action_space.index(move)
                # Create a small positive probability to avoid log(0)
                safe_prob = 1e-6 + 1e-6 * policy_logits[0].detach().clone()
                safe_prob[action_index] = 1.0 - safe_prob.sum() + safe_prob[action_index]
                log_prob = torch.log(safe_prob[action_index]).to(self.device)
                
                # Safety check for NaN
                if torch.isnan(log_prob):
                    log_prob = torch.tensor([-10.0], device=self.device)  # Safe fallback
                    
            except ValueError:
                # Move not in action space
                log_prob = torch.tensor([-10.0], device=self.device)  # Safe fallback 
                
        else:
            # Apply temperature to logits for better exploration
            temperature = 1.0
            scaled_logits = policy_logits[0] / temperature
            
            # Apply mask for legal moves
            # Create a mask for legal moves
            legal_move_mask = torch.zeros_like(scaled_logits)
            for move in self.board.legal_moves:
                try:
                    idx = self.action_space.index(move)
                    legal_move_mask[idx] = 1.0
                except ValueError:
                    continue
                    
            # Apply legal move mask (set illegal moves to large negative value)
            if legal_move_mask.sum() > 0:  # At least one legal move in the action space
                scaled_logits = scaled_logits * legal_move_mask + (1 - legal_move_mask) * -1e9
                
                # Create categorical distribution
                # Use softmax with numerical stability
                max_logit = scaled_logits.max()
                exp_logits = torch.exp(scaled_logits - max_logit)
                probs = exp_logits / (exp_logits.sum() + 1e-9)
                
                # Check for NaN or zero probs
                if torch.isnan(probs).any() or (probs.sum() < 1e-6):
                    # Fall back to uniform distribution over legal moves
                    probs = legal_move_mask / (legal_move_mask.sum() + 1e-9)
                
                # Sample from the distribution
                m = torch.distributions.Categorical(probs)
                action = m.sample()
                action_index = action.item()
                
                # Get the corresponding move
                move = self.action_space[action_index]
                
                # Compute log probability safely
                log_prob = m.log_prob(action).to(self.device)
                
                # Safety check for NaN
                if torch.isnan(log_prob):
                    log_prob = torch.tensor([-10.0], device=self.device)  # Safe fallback
            else:
                # No legal moves in action space - shouldn't happen but handle it
                legal_moves = list(self.board.legal_moves)
                move = random.choice(legal_moves)
                log_prob = torch.tensor([-10.0], device=self.device)  # Safe fallback
        
        return move, log_prob, pos_value
    
    def load_state_dict(self, state_dict, strict=False):
        # First try normal loading with strict=False
        super().load_state_dict(state_dict, strict=False)
        
        # Check if evaluator parameters were loaded
        eval_params_loaded = any('evaluator' in k for k in state_dict.keys())
        
        # If no evaluator parameters were loaded and we need them, 
        # create a fresh evaluator
        if not eval_params_loaded and hasattr(self, 'evaluator') and self.evaluator is not None:
            print("Warning: No evaluator parameters found in state dict. Using default evaluator.")


    def reinforce_update(self, optimizer, game_histories, gamma=0.99, pos_alignment_weight=0.01):
        policy_losses = []
        pos_alignment_losses = []

        # Process each game individually.
        for game_idx, game in enumerate(game_histories):
            if len(game) == 0:
                continue

            # Extract log probabilities for each move - ensure they're detached from previous computation
            log_probs = []
            for move in game:
                # Make a fresh copy on the correct device
                log_prob = move["policy_info"].squeeze().clone().detach().requires_grad_(True).to(self.device)
                log_probs.append(log_prob)
            
            log_probs_tensor = torch.stack(log_probs)
            
            # Check for NaN values in log_probs
            if torch.isnan(log_probs_tensor).any():
                continue  # Skip this game if there are NaNs
                
            # Extract rewards
            rewards = [move["reward"] for move in game]
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)

            if rewards_tensor.numel() == 0:
                continue

            # Create states from scratch to avoid any reference issues
            states = []
            for move in game:
                state = move["state"].clone().detach().to(self.device)
                states.append(state)
            
            # Forward pass through the evaluator for each state
            pos_values = []
            for state_idx, state in enumerate(states):
                try:
                    with torch.enable_grad():
                        pos_value = self.evaluator(state, self.color)
                        pos_values.append(pos_value)
                except Exception as e:
                    return 0.0  # Early exit on error
            
            pos_values_tensor = torch.stack(pos_values)
            
            # Check for NaN in positional values
            if torch.isnan(pos_values_tensor).any():
                continue
            
            # Create discount factors safely
            discounts = torch.tensor([gamma ** i for i in range(len(rewards_tensor))], 
                                    dtype=torch.float32, device=self.device)
            
            # Compute discounted rewards and returns
            discounted_rewards = rewards_tensor * discounts
            
            # Use a safe cumulative sum by creating a fresh tensor
            flipped_rewards = torch.flip(discounted_rewards, dims=[0])
            cumsum = torch.zeros_like(flipped_rewards)
            for i in range(len(flipped_rewards)):
                if i == 0:
                    cumsum[i] = flipped_rewards[i]
                else:
                    cumsum[i] = cumsum[i-1] + flipped_rewards[i]
            
            returns = torch.flip(cumsum, dims=[0])
            
            # Check for NaN in returns
            if torch.isnan(returns).any():
                continue
            
            # Normalize returns carefully
            mean_val = returns.mean().item()
            std_val = max(returns.std().item(), 1e-9)  # Prevent division by zero
            normalized_returns = torch.zeros_like(returns)
            for i in range(len(returns)):
                normalized_returns[i] = (returns[i].item() - mean_val) / std_val
            
            # Check for NaN in normalized returns
            if torch.isnan(normalized_returns).any():
                continue
            
            # Compute policy gradient loss carefully
            products = -log_probs_tensor * normalized_returns
            if torch.isnan(products).any():
                continue
                
            policy_loss = products.mean()
            policy_losses.append(policy_loss)
            
            # Compute alignment loss
            try:
                pos_alignment_loss = F.mse_loss(pos_values_tensor, normalized_returns)
                pos_alignment_losses.append(pos_alignment_loss)
            except Exception:
                continue

        if len(policy_losses) == 0:
            return 0.0

        # Average the losses carefully
        avg_policy_loss = torch.stack(policy_losses).mean()
        avg_pos_alignment_loss = torch.stack(pos_alignment_losses).mean() if pos_alignment_losses else torch.tensor(0.0, device=self.device)
        
        # Combine losses
        total_loss = avg_policy_loss + pos_alignment_weight * avg_pos_alignment_loss

        optimizer.zero_grad()
        
        # Add gradient clipping
        parameters = [p for p in self.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=0.5)
        
        try:
            total_loss.backward()
            
            # Check for NaN gradients after backward
            has_nan_grad = False
            for name, param in self.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break
                    
            if not has_nan_grad:
                optimizer.step()
            
        except Exception:
            return 0.0

        # Only print the final result
        print(f"REINFORCE policy loss: {avg_policy_loss.item():.4f}, "
            f"Position alignment loss: {avg_pos_alignment_loss.item():.4f}")

        return total_loss.item()

# class ChessHybridNet(nn.Module, ChessRL):
#     def __init__(self, board, color, device, layers=2, epsilon=0.1, evaluator=None):
#         """
#         If layers == 2, the network uses the original two conv layers:
#           - conv1: 12 -> 32 channels
#           - conv2: 32 -> 64 channels
#         If layers > 2, the network will have one initial layer (12->32),
#         then (layers-2) additional layers maintaining 32 channels,
#         and finally one layer mapping 32 -> 64 channels.
#         """
#         nn.Module.__init__(self)
#         ChessRL.__init__(self, board, color)
#         self.device = device
#         self.epsilon = epsilon

#         self.evaluator = evaluator if evaluator is not None else ChessPositionalEvaluator(device)

#         conv_layers = []
#         # First layer: convert input channels (12) to 128
#         conv_layers.append(nn.Conv2d(in_channels=12, out_channels=128, kernel_size=3, padding=1))
#         conv_layers.append(nn.ReLU())
        
#         if layers > 2:
#             # Add (layers - 2) intermediate layers maintaining 128 channels
#             for _ in range(layers - 2):
#                 conv_layers.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1))
#                 conv_layers.append(nn.ReLU())
        
#         # Keep the last layer at 128 channels as well
#         if layers > 1:
#             conv_layers.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1))
#             conv_layers.append(nn.ReLU())
        
#         self.conv_layers = nn.Sequential(*conv_layers)
        
#         # After the conv layers, the spatial dimensions remain 8x8
#         self.fc1 = nn.Linear(128 * 8 * 8, 512)
#         self.relu = nn.ReLU()
        
#         num_actions = len(ChessRL.action_space)
#         self.policy_head = nn.Linear(512, num_actions)
#         self.value_head = nn.Linear(512, 1)


#     def choose_move(self, num_simulations=10):
#         best_move, policy_info = mcts_search(self.board, self, self.action_space, num_simulations)
#         return best_move, policy_info

#     def forward(self, board_tensor):
#         # If the input is unbatched (shape: [12, 8, 8]), add a batch dimension.
#         if board_tensor.ndim == 3:
#             x = board_tensor.to(self.device).unsqueeze(0)  # Now shape: (1, 12, 8, 8)
#         else:
#             # Assume the input is already batched (shape: [N, 12, 8, 8]).
#             x = board_tensor.to(self.device)
        
#         x = self.conv_layers(x)        # Process through conv layers.
#         x = x.view(x.size(0), -1)        # Flatten: shape becomes (N, 64*8*8)
#         x = F.relu(self.fc1(x))
#         policy_logits = self.policy_head(x)
#         value = torch.tanh(self.value_head(x))  # Value in range [-1,1]
#         return F.softmax(policy_logits, dim=-1), value


#     def reinforce_update(self, optimizer, game_histories):
#         """
#         Perform an update on the hybrid network using self-play samples.
#         Each move dictionary in game_histories should have:
#         - "state": a board tensor (e.g. [12,8,8])
#         - "policy_info": a target policy distribution (numpy array of shape [num_actions])
#         - "target_value": the final outcome from the perspective of the moving agent.
        
#         Returns:
#         The average loss computed over the batch.
#         """
#         losses = []
        
#         for game in game_histories:
#             if len(game) == 0:
#                 continue

#             for move in game:
#                 state = move["state"]  # Assume shape [12, 8, 8]
#                 # Forward pass: expected output shapes: 
#                 # predicted_policy: (1, num_actions) and predicted_value: (1, 1)
#                 predicted_policy, predicted_value = self.forward(state)
                
#                 # Convert target policy distribution to tensor (shape: (1, num_actions)).
#                 target_policy = torch.tensor(move["policy_info"],
#                                             dtype=torch.float32,
#                                             device=self.device).unsqueeze(0)
                
#                 # Convert target value to tensor (shape: (1,)).
#                 target_value = torch.tensor([move["target_value"]],
#                                             dtype=torch.float32,
#                                             device=self.device)
                
#                 # Policy loss: equivalent to -sum(target_policy * log(predicted_policy)).
#                 policy_loss = -torch.sum(target_policy * torch.log(predicted_policy + 1e-8))
                
#                 # Value loss: squared error between predicted value and target value.
#                 value_loss = (target_value - predicted_value.squeeze()) ** 2
                
#                 loss = policy_loss + value_loss
#                 losses.append(loss)
        
#         if not losses:
#             print("No move history found! Skipping update.")
#             return 0.0

#         total_loss = torch.stack(losses).mean()
#         optimizer.zero_grad()
#         total_loss.backward()
#         optimizer.step()

#         print(f"REINFORCE loss: {total_loss.item():.4f}")
#         return total_loss.item()
