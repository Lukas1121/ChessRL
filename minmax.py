import chess

def minimax(board, depth, maximizing, eval_func):
    """
    A basic minimax implementation.

    Args:
        board (chess.Board): The current board state.
        depth (int): The depth to search.
        maximizing (bool): True if the current layer is maximizing, False if minimizing.
        eval_func (function): A function that evaluates the board (e.g., your compute_material_score).

    Returns:
        float: The evaluation score for the board.
    """
    if depth == 0 or board.is_game_over():
        return eval_func(board)

    if maximizing:
        max_eval = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval_val = minimax(board, depth - 1, False, eval_func)
            board.pop()
            max_eval = max(max_eval, eval_val)
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval_val = minimax(board, depth - 1, True, eval_func)
            board.pop()
            min_eval = min(min_eval, eval_val)
        return min_eval

def get_best_move(board, depth, eval_func):
    """
    Return the best move for the current board state using minimax.

    Args:
        board (chess.Board): The current board state.
        depth (int): The search depth.
        eval_func (function): Evaluation function to assess board positions.

    Returns:
        chess.Move: The best move found.
    """
    best_move = None
    best_eval = -float('inf') if board.turn == chess.WHITE else float('inf')
    
    for move in board.legal_moves:
        board.push(move)
        eval_val = minimax(board, depth - 1, board.turn == chess.BLACK, eval_func)
        board.pop()
        
        if board.turn == chess.WHITE:
            if eval_val > best_eval:
                best_eval = eval_val
                best_move = move
        else:
            if eval_val < best_eval:
                best_eval = eval_val
                best_move = move
    return best_move
