# Helper_functions.py

import os
import chess
from ChessRL import ChessRL, ChessPolicyNet, reinforce_update
import torch
import random
import os
import numpy as np
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use the first available GPU
    print("GPU is available and being used.")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU instead.")
import matplotlib.pyplot as plt

# Now you can define your helper functions here
def play_self_game(policy_net_white, policy_net_black, terminal_reward=100, terminal_loss=100,
                   per_move_penalty=0, move_length_threshold=200, exceeding_length_penalty=100):
    """
    Play a self-play game between two policy networks, forcefully ending the game
    if it exceeds move_length_threshold, and adjust terminal rewards accordingly.
    
    Returns:
        tuple: (white_move_history, black_move_history, game_length, result)
    """
    board = chess.Board()
    
    while not board.is_game_over():
        if len(policy_net_white.move_history) >= move_length_threshold:
            break

        if board.turn == chess.WHITE:
            policy_net_white.update_board(board)
            move = policy_net_white.choose_move()
            if move not in board.legal_moves:
                continue

            # Check if the move is a capture
            is_capture = board.is_capture(move)

            board.push(move)
            policy_net_white.update_board(board)

            if not is_capture:
                policy_net_white.move_history[-1]['points'] += policy_net_white.non_capture_penalty
        else:
            policy_net_black.update_board(board)
            move = policy_net_black.choose_move()
            if move not in board.legal_moves:
                continue

            # Check if the move is a capture
            is_capture = board.is_capture(move)

            board.push(move)
            policy_net_black.update_board(board)

            # Apply penalty if no capture occurred
            if not is_capture:
                policy_net_black.move_history[-1]['points'] += policy_net_black.non_capture_penalty

    result = board.result()
    
    game_length = len(policy_net_white.move_history)
    length_penalty = per_move_penalty * game_length

    if result == "1-0":
        terminal_reward_white = terminal_reward
        terminal_reward_black = -terminal_loss
    elif result == "0-1":
        terminal_reward_white = -terminal_loss
        terminal_reward_black = terminal_reward
    else:
        terminal_reward_white = -length_penalty
        terminal_reward_black = -length_penalty

    if game_length >= move_length_threshold:
        terminal_reward_white -= exceeding_length_penalty
        terminal_reward_black -= exceeding_length_penalty

    if policy_net_white.move_history:
        policy_net_white.move_history[-1]['points'] = policy_net_white.move_history[-1].get('points', 0) + terminal_reward_white
    if policy_net_black.move_history:
        policy_net_black.move_history[-1]['points'] = policy_net_black.move_history[-1].get('points', 0) + terminal_reward_black

    return policy_net_white.move_history, policy_net_black.move_history, game_length, result

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
    move_length_threshold=200,
    exceeding_length_penalty=1000,
    pretrained_model_path_white=None,
    pretrained_model_path_black=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    Train chess policy networks using self-play and the REINFORCE algorithm.
    (Documentation omitted for brevity)
    """
    import chess  # Ensure chess is imported if not already
    # Instantiate networks using classes from ChessRL.py
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

    optimizer_white = torch.optim.Adam(policy_net_white.parameters(), lr=lr)
    optimizer_black = torch.optim.Adam(policy_net_black.parameters(), lr=lr)

    white_avg_points = []
    black_avg_points = []
    white_loss_list = []
    black_loss_list = []
    white_avg_log_probs = []
    black_avg_log_probs = []
    game_length_list = []
    white_win_rates = []
    black_win_rates = []
    draw_rates = []

    for iteration in range(num_iterations):
        epsilon = max(epsilon_final, epsilon_initial - (epsilon_initial - epsilon_final) * (iteration / num_iterations))
        policy_net_white.epsilon = epsilon
        policy_net_black.epsilon = epsilon

        print(f"Starting iteration {iteration+1}/{num_iterations} with epsilon = {epsilon:.4f}")

        batch_white_move_history = []
        batch_black_move_history = []
        game_lengths = []
        results = []

        for _ in range(games_per_iteration):
            wm_history, bm_history, game_length, result = play_self_game(
                policy_net_white, policy_net_black,
                per_move_penalty=per_move_penalty,
                terminal_reward=terminal_reward,
                terminal_loss=terminal_loss,
                move_length_threshold=move_length_threshold,
                exceeding_length_penalty=exceeding_length_penalty
            )
            batch_white_move_history.extend(wm_history)
            batch_black_move_history.extend(bm_history)
            game_lengths.append(game_length)
            results.append(result)

        policy_net_white.move_history = batch_white_move_history
        policy_net_black.move_history = batch_black_move_history

        white_loss = reinforce_update(policy_net_white, optimizer_white, gamma=gamma)
        black_loss = reinforce_update(policy_net_black, optimizer_black, gamma=gamma)
        white_loss_list.append(white_loss)
        black_loss_list.append(black_loss)

        white_avg = np.mean([entry['points'] for entry in batch_white_move_history])
        black_avg = np.mean([entry['points'] for entry in batch_black_move_history])
        white_avg_points.append(white_avg)
        black_avg_points.append(black_avg)


        avg_game_length = np.mean(game_lengths)
        game_length_list.append(avg_game_length)

        white_wins = sum(1 for r in results if r == "1-0")
        draws = sum(1 for r in results if r == "1/2-1/2")
        white_losses = sum(1 for r in results if r == "0-1")
        total_games = len(results)
        win_rate_white = white_wins / total_games
        draw_rate = draws / total_games
        win_rate_black = white_losses / total_games
        white_win_rates.append(win_rate_white)
        black_win_rates.append(win_rate_black)
        draw_rates.append(draw_rate)

        print(f"Iteration {iteration+1} complete.")
        print(f"  Avg Game Length: {avg_game_length:.2f}")
        print(f"  White Avg Points: {white_avg:.2f}, Black Avg Points: {black_avg:.2f}")
        print(f"  White Win Rate: {win_rate_white:.2f}, Black Win Rate: {win_rate_black:.2f}, Draw Rate: {draw_rate:.2f}\n")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    run_num = 1
    while os.path.exists(os.path.join(base_dir, f"run{run_num}")):
        run_num += 1
    save_folder = os.path.join(base_dir, f"run{run_num}")
    os.makedirs(save_folder, exist_ok=True)

    white_save_path = os.path.join(save_folder, "policy_net_white.pth")
    black_save_path = os.path.join(save_folder, "policy_net_black.pth")

    torch.save(policy_net_white.state_dict(), white_save_path)
    torch.save(policy_net_black.state_dict(), black_save_path)
    print("Models saved in folder:", save_folder)

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
        "device": str(device)
    }

    params_file = os.path.join(save_folder, "parameters.txt")
    with open(params_file, "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    print("Parameters saved to:", params_file)

    metrics = {
        "policy_net_white": policy_net_white,
        "policy_net_black": policy_net_black,
        "white_loss_list": white_loss_list,
        "black_loss_list": black_loss_list,
        "white_avg_points": white_avg_points,
        "black_avg_points": black_avg_points,
        "white_avg_log_probs": white_avg_log_probs,
        "black_avg_log_probs": black_avg_log_probs,
        "game_length_list": game_length_list,
        "white_win_rates": white_win_rates,
        "black_win_rates": black_win_rates,
        "draw_rates": draw_rates
    }

    return metrics


def bin_data(data, num_bins):
    """
    Splits a list/array of data into a specified number of bins,
    and computes the mean and standard deviation in each bin.
    
    Parameters:
        data (list or np.array): The data to be binned.
        num_bins (int): Number of bins to aggregate the data into.
    
    Returns:
        bin_centers (np.array): The x-axis positions (bin centers or representative iteration numbers).
        bin_means (np.array): The mean value in each bin.
        bin_stds (np.array): The standard deviation in each bin.
    """
    data = np.array(data)
    n = len(data)
    bin_size = int(np.ceil(n / num_bins))
    bin_means = []
    bin_stds = []
    bin_centers = []
    for i in range(0, n, bin_size):
        bin_slice = data[i:i+bin_size]
        bin_means.append(np.mean(bin_slice))
        bin_stds.append(np.std(bin_slice))
        # Use the midpoint of the bin as the representative iteration number:
        bin_centers.append(i + len(bin_slice) / 2.0)
    return np.array(bin_centers), np.array(bin_means), np.array(bin_stds)

def plot_training_metrics_binned(metrics, num_bins=20):
    """
    Plot training metrics by binning the iteration data to provide a clearer view of trends.
    
    Parameters:
        metrics (dict): Dictionary containing the following keys:
            - white_loss_list
            - black_loss_list
            - white_avg_points
            - black_avg_points
            - white_avg_log_probs
            - black_avg_log_probs
            - game_length_list
            - white_win_rates
            - black_win_rates
            - draw_rates
        num_bins (int): Number of bins into which the iterations will be grouped.
    """
    # Set up a 2x3 grid for plotting.
    fig, axs = plt.subplots(2, 2, figsize=(18, 10))

    # 1. Binned REINFORCE Loss for White and Black.
    iterations, white_loss_means, white_loss_stds = bin_data(metrics['white_loss_list'], num_bins)
    _, black_loss_means, black_loss_stds = bin_data(metrics['black_loss_list'], num_bins)
    axs[0, 0].errorbar(iterations, white_loss_means, yerr=white_loss_stds, marker='o', color='blue', label='White Loss')
    axs[0, 0].errorbar(iterations, black_loss_means, yerr=black_loss_stds, marker='o', color='red', label='Black Loss')
    axs[0, 0].set_xlabel("Iteration")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].set_title("Binned REINFORCE Loss per Iteration")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. Binned Average Points per Iteration.
    iterations, white_avg_points_means, white_avg_points_stds = bin_data(metrics['white_avg_points'], num_bins)
    _, black_avg_points_means, black_avg_points_stds = bin_data(metrics['black_avg_points'], num_bins)
    axs[0, 1].errorbar(iterations, white_avg_points_means, yerr=white_avg_points_stds, marker='o', color='blue', label='White Avg Points')
    axs[0, 1].errorbar(iterations, black_avg_points_means, yerr=black_avg_points_stds, marker='o', color='red', label='Black Avg Points')
    axs[0, 1].set_xlabel("Iteration")
    axs[0, 1].set_ylabel("Avg Points")
    axs[0, 1].set_title("Binned Average Points per Iteration")
    axs[0, 1].legend()
    axs[0, 1].grid(True)


    # 5. Binned Average Game Length per Iteration.
    iterations, game_length_means, game_length_stds = bin_data(metrics['game_length_list'], num_bins)
    axs[1, 1].errorbar(iterations, game_length_means, yerr=game_length_stds, marker='o', color='purple')
    axs[1, 1].set_xlabel("Iteration")
    axs[1, 1].set_ylabel("Moves")
    axs[1, 1].set_title("Binned Average Game Length per Iteration")
    axs[1, 1].grid(True)

    # 6. Binned Win/Loss/Draw Rates per Iteration.
    iterations, white_win_rates_means, white_win_rates_stds = bin_data(metrics['white_win_rates'], num_bins)
    _, black_win_rates_means, black_win_rates_stds = bin_data(metrics['black_win_rates'], num_bins)
    _, draw_rates_means, draw_rates_stds = bin_data(metrics['draw_rates'], num_bins)
    axs[1, 0].errorbar(iterations, white_win_rates_means, yerr=white_win_rates_stds, marker='o', color='blue', label='White Win Rate')
    axs[1, 0].errorbar(iterations, black_win_rates_means, yerr=black_win_rates_stds, marker='o', color='red', label='Black Win Rate')
    axs[1, 0].errorbar(iterations, draw_rates_means, yerr=draw_rates_stds, marker='o', color='green', label='Draw Rate')
    axs[1, 0].set_xlabel("Iteration")
    axs[1, 0].set_ylabel("Rate")
    axs[1, 0].set_title("Binned Win/Loss/Draw Rates per Iteration")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    plt.tight_layout()
    plt.show()

def print_custom_board(board):
    """
    Prints the chess board with a fixed-width for each square.
    Each piece is shown with a preceding 'white ' for white pieces and 'black ' for black pieces,
    centered in an 8-character wide field.
    Empty squares are shown as '.'.
    The board is printed from rank 8 to 1.
    """
    cell_width = 8  # Increase the width to accommodate "white " or "black " + piece symbol

    # Loop over the ranks from 8 down to 1.
    for rank in range(7, -1, -1):
        # Create the row starting with the rank number and two spaces.
        row_str = f"{rank+1}  "
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            if piece is None:
                cell_str = "."
            else:
                # Prepend with 'white ' or 'black ' and append the piece symbol (upper-case)
                if piece.color == chess.WHITE:
                    cell_str = f"white {piece.symbol().upper()}"
                else:
                    cell_str = f"black {piece.symbol().upper()}"
            # Center the cell string in a field of width cell_width
            row_str += f"{cell_str:^{cell_width}}"
        print(row_str)

    # Compute the left margin based on the rank label (e.g. "8  " is 3 characters).
    label_width = len(f"{8}  ")
    file_str = " " * label_width
    for file in range(8):
        file_letter = chr(ord('a') + file)
        file_str += f"{file_letter:^{cell_width}}"
    print(file_str)
    print()


def play_human_vs_bot(white_model_path, black_model_path, human_color=None,
                      use_lookahead=False, minimax_depth=1,top_n=3):
    """
    Play a game of chess between a human and the bot.
    
    Parameters:
        white_model_path (str): Path to the pretrained model file for White.
        black_model_path (str): Path to the pretrained model file for Black.
        human_color (chess.WHITE or chess.BLACK or None): The color you wish to play.
            If set to None, the bot's color is chosen at random, and you play the opposite.
        device (torch.device): The device on which to load the model.
        use_minimax (bool): If True, the bot will use minimax search to select its moves.
        minimax_depth (int): The depth to search when using minimax.
        
    Behavior:
        - If human_color is provided, the bot plays the opposite color.
        - If human_color is None, a random color is chosen for the bot.
    
    Assumes:
        - A function `print_custom_board(board)` exists for displaying the board.
        - The classes `ChessRL` and `ChessPolicyNet` have been defined and imported.
    
    Returns:
        None
    """
    board = chess.Board()
    
    # Determine colors.
    if human_color is None:
        bot_color = random.choice([chess.WHITE, chess.BLACK])
        human_color = not bot_color
    else:
        bot_color = not human_color

    print("Bot plays as:", "White" if bot_color == chess.WHITE else "Black")
    print("You play as:", "White" if human_color == chess.WHITE else "Black")
    print()

    # Create the policy network for the bot and load the appropriate model.
    if bot_color == chess.WHITE:
        policy_net = ChessPolicyNet(
            num_actions=len(ChessRL.action_space),
            board=board,
            color=chess.WHITE
        ).to(device)
        policy_net.load_state_dict(torch.load(white_model_path, map_location=device))
    else:
        policy_net = ChessPolicyNet(
            num_actions=len(ChessRL.action_space),
            board=board,
            color=chess.BLACK
        ).to(device)
        policy_net.load_state_dict(torch.load(black_model_path, map_location=device))
    
    # Set the model to evaluation mode.
    policy_net.eval()

    # Game loop.
    while not board.is_game_over():
        # Display the current board.
        print_custom_board(board)
        
        if board.turn == bot_color:
            # Bot's turn.
            policy_net.update_board(board)
            if use_lookahead:
                # Use the minimax branch in choose_move.
                move = policy_net.choose_move(use_lookahead=use_lookahead, minimax_depth=minimax_depth,top_n=top_n)
            else:
                # Use the default RL-based move selection.
                move = policy_net.choose_move()
            
            if move not in board.legal_moves:
                print("Selected move is illegal! (Something is wrong with masking.)")
                continue
            
            print("Bot plays:", move)
            board.push(move)
        else:
            # Human's turn.
            move_str = input("Your move (in UCI format, e.g., e2e4): ").strip()
            try:
                move = chess.Move.from_uci(move_str)
            except ValueError:
                print("Invalid move format. Please try again.\n")
                continue
            
            if move in board.legal_moves:
                board.push(move)
            else:
                print("Illegal move. Please try again.\n")
                continue
        
        print()  # Blank line for readability.
    
    # Final board display and result.
    print_custom_board(board)
    print("Game over!")
    print("Result:", board.result())
