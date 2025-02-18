import chess
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

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
        chess.KING: 100,
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
    
    def compute_material_score(self):
        # Compute the score using vectorized multiplication.
        total_score = torch.sum(self.board_tensor * self.value_tensor)
        # Return the score from the perspective of the agent's color.
        return total_score if self.color == chess.WHITE else -total_score       

    def get_positional_value(self, piece_type, row, col):
        if piece_type == chess.PAWN:
            return self.pawn_table[row, col]
        elif piece_type == chess.KNIGHT:
            return self.knight_table[row, col]
        elif piece_type == chess.BISHOP:
            return self.bishop_table[row, col]
        elif piece_type == chess.ROOK:
            return self.rook_table[row, col]
        elif piece_type == chess.QUEEN:
            return self.queen_table[row, col]
        elif piece_type == chess.KING:
            return self.king_table[row, col]
        return 0


    def update_board(self, board):
        self.board = board
        self.board_tensor = self.board_to_tensor(board)
        current_score = self.compute_material_score()
        # Compute the material difference from the previous state.
        self.material_delta = current_score - self.last_material_score
        # Update the last_material_score for the next comparison.
        self.last_material_score = current_score

    def _save_move(self, move, action_index, log_prob, points):
        """Stores move data efficiently using tensors."""
        reward_tensor = torch.tensor([points], dtype=torch.float32, device=self.device).detach()

        # Append move data as a single tensor batch
        if not hasattr(self, "move_history_tensors"):
            self.move_history_tensors = {
                "log_probs": torch.empty(0, dtype=torch.float32, device=self.device),
                "rewards": torch.empty(0, dtype=torch.float32, device=self.device),
                "action_indices": torch.empty(0, dtype=torch.long, device=self.device),
            }

        self.move_history_tensors["log_probs"] = torch.cat([self.move_history_tensors["log_probs"], log_prob.view(1)])
        self.move_history_tensors["rewards"] = torch.cat([self.move_history_tensors["rewards"], reward_tensor])
        self.move_history_tensors["action_indices"] = torch.cat([self.move_history_tensors["action_indices"], torch.tensor([action_index], dtype=torch.long, device=self.device)])

    def clear_move_history(self):
        self.move_history_tensors = {
            "log_probs": torch.empty(0, dtype=torch.float32, device=self.device),
            "rewards": torch.empty(0, dtype=torch.float32, device=self.device),
            "action_indices": torch.empty(0, dtype=torch.long, device=self.device),
        }

class ChessPolicyNet(nn.Module, ChessRL):
    def __init__(self, board, color, device, non_capture_penalty=-0.2, epsilon=0.1):
        nn.Module.__init__(self)
        ChessRL.__init__(self, board, color)
        # Define convolutional layers:
        self.device = device
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        num_actions = len(ChessRL.action_space)

        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_actions)

        self.non_capture_penalty = non_capture_penalty
        self.epsilon = epsilon  # Exploration parameter.

        self.move_history = []
        self.last_material_score = self.compute_material_score()

    def forward(self):
        # Use the stored board tensor; assume it’s already updated.
        # Add batch dimension if necessary.
        x = self.board_tensor.to(self.device).unsqueeze(0)  # Now x has shape (1, 12, 8, 8)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        # Generate legal mask and apply it.
        legal_mask = self.create_legal_mask().to(self.device)  # Shape: (num_actions,)
        legal_mask = legal_mask.unsqueeze(0).expand_as(logits)
        masked_logits = logits + (legal_mask - 1) * 1e8
        probs = F.softmax(masked_logits, dim=-1)
        return probs

    def _minimax(self, board, depth, alpha, beta, maximizing):
        """
        Minimax search with alpha-beta pruning.

        Args:
            board (chess.Board): The board state to evaluate.
            depth (int): How many plies to search.
            alpha (float): The best already explored option along the path to the root for the maximizer.
            beta (float): The best already explored option along the path to the root for the minimizer.
            maximizing (bool): True if the current node is a maximizing node.

        Returns:
            float: The evaluation score.
        """
        # Terminal condition: depth 0 or game over.
        if depth == 0 or board.is_game_over():
            return self.compute_material_score()

        if maximizing:
            max_eval = -float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval_value = self._minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval_value)
                alpha = max(alpha, eval_value)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval_value = self._minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval_value)
                beta = min(beta, eval_value)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval

    def choose_move(self, method="rl", minmax_depth=2, top_n=3, simulations=100):
        """
        Selects a move using one of three methods:
        - "lookahead": Uses RL with Minimax for deeper search.
        - "mcts": Uses Monte Carlo Tree Search.
        - "rl": Uses the RL network with epsilon-greedy.

        Args:
            method (str): "lookahead", "mcts", or "rl".
            minmax_depth (int): Depth of the Minimax search (if using lookahead).
            top_n (int): Number of candidate moves to evaluate (for Minimax).
            simulations (int): Number of simulations for MCTS.

        Returns:
            chess.Move: The selected move.
        """
        if method == "lookahead":
            print("lookahead")
            return self._choose_with_lookahead(minmax_depth, top_n)
        elif method == "mcts":
            return mcts(self.board, simulations)
        else:  # Default: RL with epsilon-greedy
            return self._choose_with_rl()

    def _choose_with_lookahead(self, minmax_depth, top_n):
        """Selects a move using RL-based move probabilities combined with Minimax lookahead."""
        probs = self.forward()  # Get action probabilities
        probs_np = probs.cpu().detach().numpy().flatten()
        top_indices = probs_np.argsort()[-top_n:][::-1]  # Select top N moves

        candidate_moves = []
        candidate_indices = []
        for idx in top_indices:
            move = self.action_space[idx]
            if move in self.board.legal_moves:
                candidate_moves.append(move)
                candidate_indices.append(idx)
        if not candidate_moves:
            candidate_moves = list(self.board.legal_moves)
            candidate_indices = [self.action_space.index(move) for move in candidate_moves]

        best_move, best_eval, best_index = None, None, None
        for idx, move in zip(candidate_indices, candidate_moves):
            self.board.push(move)
            eval_value = self._minimax(self.board, minmax_depth, -float('inf'), float('inf'), maximizing=(self.board.turn == self.color))
            self.board.pop()
            if best_move is None or (self.color == chess.WHITE and eval_value > best_eval) or (self.color == chess.BLACK and eval_value < best_eval):
                best_move, best_eval, best_index = move, eval_value, idx

        log_prob = torch.distributions.Categorical(probs).log_prob(torch.tensor(best_index, device=self.device))
        points = self.compute_material_score()
        self._save_move(best_move, best_index, log_prob, points, lookahead=True)
        return best_move

    def _choose_with_rl(self):
        """Selects a move using epsilon-greedy RL policy."""
        if np.random.rand() < self.epsilon:
            legal_moves = list(self.board.legal_moves)
            move = random.choice(legal_moves)
            action_index = self.action_space.index(move)
            dummy_log_prob = torch.tensor(0.0, device=self.device)
            points = self.compute_material_score()
            self._save_move(move, action_index, dummy_log_prob, points)
            return move

        probs = self.forward()
        m = D.Categorical(probs)
        action = m.sample()
        action_index = action.item()
        move = self.action_space[action_index]

        log_prob = m.log_prob(action).to(self.device)
        points = self.compute_material_score()
        self._save_move(move, action_index, log_prob, points)
        return move

    def reinforce_update(self, optimizer, game_histories, gamma=0.99):
        """
        Perform a REINFORCE update using a batch of game histories.

        Args:
            optimizer: The optimizer for updating the network.
            game_histories: A list of dictionaries, one per game, each containing:
                - "log_probs": Tensor of shape [num_moves]
                - "rewards": Tensor of shape [num_moves]
                - "action_indices": (optional) Tensor of shape [num_moves]
            gamma: Discount factor.

        Returns:
            The average loss (float) computed over the batch.
        """
        losses = []

        # Process each game individually
        for history in game_histories:
            log_probs = history["log_probs"]  # shape: [n_moves]
            rewards = history["rewards"]      # shape: [n_moves]

            # Skip if empty history
            if rewards.numel() == 0:
                continue

            # Create discount factors [1, gamma, gamma^2, ...]
            discounts = torch.tensor([gamma**i for i in range(len(rewards))],
                                    dtype=torch.float32,
                                    device=self.device)

            # Compute discounted returns:
            # First, multiply rewards by discount factors,
            # then compute the cumulative sum in reverse order,
            # and finally flip back.
            returns = torch.flip(torch.cumsum(torch.flip(rewards * discounts, dims=[0]), dim=0), dims=[0])

            # Normalize returns for this game
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

            # Compute loss for the game
            loss = (-log_probs * returns).mean()
            losses.append(loss)

        if len(losses) == 0:
            print("No move history found! Skipping update.")
            return 0.0

        # Average the losses over all games
        total_loss = torch.stack(losses).mean()

        # Perform backpropagation and update the network
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"REINFORCE loss: {total_loss.item():.4f}")
        return total_loss.item()

class ChessHybridNet(nn.Module,ChessRL):
    def __init__(self, board, color, device):
        nn.Module.__init__(self)
        ChessRL.__init__(self, board, color)
        self.device = device
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        
        num_actions = len(ChessRL.action_space)
        self.policy_head = nn.Linear(512, num_actions)
        self.value_head = nn.Linear(512, 1)
    
    def forward(self, board_tensor):
        x = board_tensor.to(self.device).unsqueeze(0)  # (1, 12, 8, 8)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))  # value in [-1,1]
        return F.softmax(policy_logits, dim=-1), value
