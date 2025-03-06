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
    def __init__(self, device='gpu'):
        super(ChessPositionalEvaluator, self).__init__()
        
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
                    end_w * self.knight_table_middlegame)  # Use middlegame for endgame too
        # ... other pieces ...
        elif piece_type == chess.KING:
            return (opening_w * self.king_table_opening + 
                    mid_w * self.king_table_middlegame + 
                    end_w * self.king_table_endgame)
    
    def forward(self, board_tensor, color):
        """Compute positional value using phase-specific tables"""
        phase_weights = self.detect_game_phase(board_tensor)
        
        value_tensor = torch.zeros_like(board_tensor)
        for channel in range(12):
            piece_type = (channel % 6) + 1
            is_white = channel < 6
            
            # Material value
            material_val = self.piece_values[piece_type - 1]
            
            # Get blended table for this piece type
            table = self.get_blended_table(piece_type, phase_weights)
            
            if is_white:
                value_tensor[channel] = material_val + table
            else:
                value_tensor[channel] = -(material_val + torch.flip(table, dims=[0]))
        
        # Compute final value
        total_value = torch.sum(board_tensor * value_tensor)
        return total_value if color == chess.WHITE else -total_value

class ChessPolicyNet(nn.Module, ChessRL):
    def __init__(self, board, color, device,layers=5, epsilon=0.1,evaluator=None):
        nn.Module.__init__(self)
        ChessRL.__init__(self, board, color)
        self.device = device
        self.epsilon = epsilon

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

    def forward(self):
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
        probs,value,pos_value = self.forward()
        if np.random.rand() < self.epsilon:
            legal_moves = list(self.board.legal_moves)
            move = random.choice(legal_moves)
            action_index = self.action_space.index(move)
            log_prob = torch.log(probs[0, action_index] + 1e-8).to(self.device)
            return move, log_prob

        m = D.Categorical(probs)
        action = m.sample()
        action_index = action.item()
        move = self.action_space[action_index]
        log_prob = m.log_prob(action).to(self.device)
        return move, log_prob


    def reinforce_update(self, optimizer, game_histories, gamma=0.99, pos_alignment_weight=0.1):
        """
        Perform a REINFORCE update using a batch of game histories.
        This updated version also trains the positional evaluator.
        
        Args:
            optimizer: The optimizer for updating the network.
            game_histories: A list where each element is a list of move dictionaries.
                            Each move dictionary should contain:
                            - "policy_info": a tensor (scalar or [1]) representing the log probability.
                            - "reward": a scalar reward for that move.
                            - "state": the board tensor at that move.
            gamma: Discount factor.
            pos_alignment_weight: Weight for the positional alignment loss term.
        
        Returns:
            The average loss (float) computed over the batch.
        """
        policy_losses = []
        pos_alignment_losses = []

        # Process each game individually.
        for game in game_histories:
            if len(game) == 0:
                continue

            # Extract log probabilities for each move.
            log_probs = [move["policy_info"].squeeze() for move in game]
            log_probs_tensor = torch.stack(log_probs)  # shape: [num_moves]
            
            # Extract rewards as a tensor.
            rewards_tensor = torch.tensor([move["reward"] for move in game],
                                        dtype=torch.float32,
                                        device=self.device)

            if rewards_tensor.numel() == 0:
                continue

            # For each state in the game, get the positional value
            states = [move["state"] for move in game]
            
            # Forward pass through the evaluator for each state
            pos_values = []
            for state in states:
                # Get the positional evaluation
                with torch.enable_grad():  # Ensure we're tracking gradients
                    pos_value = self.evaluator(state, self.color)
                    pos_values.append(pos_value)
            
            pos_values_tensor = torch.stack(pos_values)
            
            # Create discount factors: [1, gamma, gamma^2, ...]
            discounts = torch.tensor([gamma ** i for i in range(len(rewards_tensor))],
                                    dtype=torch.float32,
                                    device=self.device)
            # Compute discounted rewards and cumulative returns.
            discounted_rewards = rewards_tensor * discounts
            returns = torch.flip(torch.cumsum(torch.flip(discounted_rewards, dims=[0]), dim=0), dims=[0])
            
            # Normalize returns for this game.
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
            # Compute policy gradient loss
            policy_loss = (-log_probs_tensor * returns).mean()
            policy_losses.append(policy_loss)
            
            # Compute alignment loss between positional values and returns
            # This encourages the positional evaluator to predict expected returns
            pos_alignment_loss = F.mse_loss(pos_values_tensor, returns)
            pos_alignment_losses.append(pos_alignment_loss)

        if len(policy_losses) == 0:
            print("No move history found! Skipping update.")
            return 0.0

        # Average the losses over all games.
        avg_policy_loss = torch.stack(policy_losses).mean()
        avg_pos_alignment_loss = torch.stack(pos_alignment_losses).mean() if pos_alignment_losses else 0
        
        # Combine losses
        total_loss = avg_policy_loss + pos_alignment_weight * avg_pos_alignment_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

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
