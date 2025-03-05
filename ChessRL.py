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

class ChessPolicyNet(nn.Module, ChessRL):
    def __init__(self, board, color, device,layers=5, epsilon=0.1):
        nn.Module.__init__(self)
        ChessRL.__init__(self, board, color)
        self.device = device
        
        # Build a sequential container for 10 conv layers.
        conv_layers = []
        # First layer: input channels 12 -> 32
        conv_layers.append(nn.Conv2d(12, 32, kernel_size=3, padding=1))
        conv_layers.append(nn.ReLU())
        # Next 9 layers: keep 32 channels throughout.
        for _ in range(layers-1):
            conv_layers.append(nn.Conv2d(32, 32, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
        self.conv_layers = nn.Sequential(*conv_layers)
        
        num_actions = len(ChessRL.action_space)
        # Adjust the fully connected layer to match the conv output: 32 channels * 8 * 8.
        self.fc1 = nn.Linear(32 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
        self.epsilon = epsilon  # Exploration parameter.


    def forward(self):
        # Assume board_tensor is up-to-date and has shape (12, 8, 8).
        x = self.board_tensor.to(self.device).unsqueeze(0)  # Shape: (1, 12, 8, 8)
        x = self.conv_layers(x)  # Pass through the 10 conv layers.
        x = x.view(x.size(0), -1)  # Flatten.
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        # Apply legal move mask.
        legal_mask = self.create_legal_mask().to(self.device)  # Shape: (num_actions,)
        legal_mask = legal_mask.unsqueeze(0).expand_as(logits)
        masked_logits = logits + (legal_mask - 1) * 1e8
        probs = F.softmax(masked_logits, dim=-1)
        return probs

    def choose_move(self):
        """Selects a move using epsilon-greedy RL policy."""
        probs = self.forward()
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


    def reinforce_update(self, optimizer, game_histories, gamma=0.99):
        """
        Perform a REINFORCE update using a batch of game histories.
        
        Args:
            optimizer: The optimizer for updating the network.
            game_histories: A list where each element is a list of move dictionaries.
                            Each move dictionary should contain:
                            - "policy_info": a tensor (scalar or [1]) representing the log probability.
                            - "reward": a scalar reward for that move.
            gamma: Discount factor.
        
        Returns:
            The average loss (float) computed over the batch.
        """
        losses = []

        # Process each game individually.
        for game in game_histories:
            if len(game) == 0:
                continue

            # Extract and fix log probabilities for each move.
            log_probs = [move["policy_info"].squeeze() for move in game]
            log_probs_tensor = torch.stack(log_probs)  # shape: [num_moves]
            
            # Extract rewards as a tensor.
            rewards_tensor = torch.tensor([move["reward"] for move in game],
                                        dtype=torch.float32,
                                        device=self.device)

            if rewards_tensor.numel() == 0:
                continue

            # Create discount factors: [1, gamma, gamma^2, ...]
            discounts = torch.tensor([gamma ** i for i in range(len(rewards_tensor))],
                                    dtype=torch.float32,
                                    device=self.device)
            # Compute discounted rewards and cumulative returns.
            discounted_rewards = rewards_tensor * discounts
            returns = torch.flip(torch.cumsum(torch.flip(discounted_rewards, dims=[0]), dim=0), dims=[0])
            
            # Normalize returns for this game.
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
            # Compute loss for this game.
            loss = (-log_probs_tensor * returns).mean()
            losses.append(loss)

        if len(losses) == 0:
            print("No move history found! Skipping update.")
            return 0.0

        # Average the losses over all games.
        total_loss = torch.stack(losses).mean()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"REINFORCE loss: {total_loss.item():.4f}")
        return total_loss.item()


class ChessHybridNet(nn.Module, ChessRL):
    def __init__(self, board, color, device, layers=2):
        """
        If layers == 2, the network uses the original two conv layers:
          - conv1: 12 -> 32 channels
          - conv2: 32 -> 64 channels
        If layers > 2, the network will have one initial layer (12->32),
        then (layers-2) additional layers maintaining 32 channels,
        and finally one layer mapping 32 -> 64 channels.
        """
        nn.Module.__init__(self)
        ChessRL.__init__(self, board, color)
        self.device = device

        conv_layers = []
        # First layer: convert input channels (12) to 32.
        conv_layers.append(nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3, padding=1))
        conv_layers.append(nn.ReLU())
        
        if layers > 2:
            # Add (layers - 2) intermediate layers with 32 channels.
            for _ in range(layers - 2):
                conv_layers.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1))
                conv_layers.append(nn.ReLU())
        
        # Final layer: convert 32 channels to 64 channels.
        conv_layers.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1))
        conv_layers.append(nn.ReLU())
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # After the conv layers, the spatial dimensions remain 8x8.
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        num_actions = len(ChessRL.action_space)
        self.policy_head = nn.Linear(512, num_actions)
        self.value_head = nn.Linear(512, 1)

    def choose_move(self, num_simulations=10):
        best_move, policy_info = mcts_search(self.board, self, self.action_space, num_simulations)
        return best_move, policy_info

    def forward(self, board_tensor):
        # If the input is unbatched (shape: [12, 8, 8]), add a batch dimension.
        if board_tensor.ndim == 3:
            x = board_tensor.to(self.device).unsqueeze(0)  # Now shape: (1, 12, 8, 8)
        else:
            # Assume the input is already batched (shape: [N, 12, 8, 8]).
            x = board_tensor.to(self.device)
        
        x = self.conv_layers(x)        # Process through conv layers.
        x = x.view(x.size(0), -1)        # Flatten: shape becomes (N, 64*8*8)
        x = F.relu(self.fc1(x))
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))  # Value in range [-1,1]
        return F.softmax(policy_logits, dim=-1), value


    def reinforce_update(self, optimizer, game_histories):
        """
        Perform an update on the hybrid network using self-play samples.
        Each move dictionary in game_histories should have:
        - "state": a board tensor (e.g. [12,8,8])
        - "policy_info": a target policy distribution (numpy array of shape [num_actions])
        - "target_value": the final outcome from the perspective of the moving agent.
        
        Returns:
        The average loss computed over the batch.
        """
        losses = []
        
        for game in game_histories:
            if len(game) == 0:
                continue

            for move in game:
                state = move["state"]  # Assume shape [12, 8, 8]
                # Forward pass: expected output shapes: 
                # predicted_policy: (1, num_actions) and predicted_value: (1, 1)
                predicted_policy, predicted_value = self.forward(state)
                
                # Convert target policy distribution to tensor (shape: (1, num_actions)).
                target_policy = torch.tensor(move["policy_info"],
                                            dtype=torch.float32,
                                            device=self.device).unsqueeze(0)
                
                # Convert target value to tensor (shape: (1,)).
                target_value = torch.tensor([move["target_value"]],
                                            dtype=torch.float32,
                                            device=self.device)
                
                # Policy loss: equivalent to -sum(target_policy * log(predicted_policy)).
                policy_loss = -torch.sum(target_policy * torch.log(predicted_policy + 1e-8))
                
                # Value loss: squared error between predicted value and target value.
                value_loss = (target_value - predicted_value.squeeze()) ** 2
                
                loss = policy_loss + value_loss
                losses.append(loss)
        
        if not losses:
            print("No move history found! Skipping update.")
            return 0.0

        total_loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"REINFORCE loss: {total_loss.item():.4f}")
        return total_loss.item()
