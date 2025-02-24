# Hf.py

import os
import chess
from ChessRL_test import ChessPolicyNet
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

def separate_and_evaluate_game_histories(game_histories, results, terminal_reward, per_move_penalty, agent_white, agent_black):
    """
    Given a list of game histories (each a list of move sample dicts), the game results,
    and a terminal reward, compute the reward for each move and separate the moves into
    white and black game histories.
    
    Each move sample is expected to contain:
      - 'state': a torch.Tensor of shape [12, 8, 8]
      - 'policy_info': the output from move selection (e.g. log_prob)
    
    For each move, an intermediate reward is computed as:
         reward = sum(state * value_tensor)
    where value_tensor is computed via the agent's precompute_value_tensor() method.
    For white moves the reward is used as-is, while for black moves its sign is flipped.
    
    Additionally, based on the final game result (from the results list) the terminal reward is applied:
      - For a white win ("1-0"): the last white move's reward is increased by terminal_reward,
        and the last black move's reward is decreased by terminal_reward.
      - For a black win ("0-1"): the last white move's reward is decreased by terminal_reward,
        and the last black move's reward is increased by terminal_reward.
      - For a draw ("1/2-1/2"): no terminal reward is applied.
      
    The final score for each game is defined as:
      - White win ("1-0"): white = terminal_reward, black = -terminal_reward.
      - Black win ("0-1"): white = -terminal_reward, black = terminal_reward.
      - Draw ("1/2-1/2"): white = 0, black = 0.
    
    Returns:
      - white_game_histories: List of game histories (one per game) for white moves.
      - black_game_histories: List of game histories (one per game) for black moves.
      - avg_white_points: The average final score for white across the batch.
      - avg_black_points: The average final score for black across the batch.
    """
    white_value_tensor = agent_white.precompute_value_tensor()
    black_value_tensor = agent_black.precompute_value_tensor()

    white_game_histories = []
    black_game_histories = []
    white_final_scores = []
    black_final_scores = []
    
    for game_index, game in enumerate(game_histories):
        white_moves = []
        black_moves = []
        # Compute intermediate rewards for each move.
        for i, sample in enumerate(game):
            if i % 2 == 0:  # White's move.
                reward = torch.sum(sample['state'] * white_value_tensor).item()
                sample['reward'] = reward
                white_moves.append(sample)
            else:           # Black's move.
                reward = -torch.sum(sample['state'] * black_value_tensor).item()
                sample['reward'] = reward
                black_moves.append(sample)
        
        # Adjust final move rewards with terminal reward.
        result = results[game_index]
        if result == "1-0":
            if white_moves:
                white_moves[-1]['reward'] += terminal_reward
            if black_moves:
                black_moves[-1]['reward'] -= terminal_reward
            white_final = terminal_reward
            black_final = -terminal_reward
        elif result == "0-1":
            if white_moves:
                white_moves[-1]['reward'] -= terminal_reward
            if black_moves:
                black_moves[-1]['reward'] += terminal_reward
            white_final = -terminal_reward
            black_final = terminal_reward
        else:  # Draw ("1/2-1/2")
            draw_penalty = -per_move_penalty * len(game)
            white_final = draw_penalty
            black_final = draw_penalty

        white_game_histories.append(white_moves)
        black_game_histories.append(black_moves)
        white_final_scores.append(white_final)
        black_final_scores.append(black_final)
    
    white_avg_points = np.mean(white_final_scores) if white_final_scores else 0
    black_avg_points = np.mean(black_final_scores) if black_final_scores else 0

    return white_game_histories, black_game_histories, white_avg_points, black_avg_points



def generate_self_play_samples(agent_white, agent_black, config):
    """
    Plays one self-play game and returns a list of training samples.
    
    Each sample is a dict with:
      - 'state': board tensor (e.g. torch.Tensor of shape [12,8,8])
      - 'target_policy': a vector of target probabilities (from MCTS visit counts)
      - 'target_value': the final outcome (+1, -1, or 0) from the perspective of the moving agent.
    """
    samples = []
    board = chess.Board()

    while not board.is_game_over():
        # Check move history or other stopping conditions if needed.
        if board.turn == chess.WHITE:
            agent = agent_white
        else:
            agent = agent_black
        
        agent.update_board(board)
        # Use the appropriate move selection method (pure RL or hybrid) based on agent configuration.
        move, policy_info = agent.choose_move()
        
        # record the current state, target policy, and log_prob
        sample = {
            "state": agent.board_tensor.clone(),
            "policy_info": policy_info,
        }
        samples.append(sample)
        
        if move not in board.legal_moves:
            continue
        
        board.push(move)
        agent.update_board(board)
    
    result = board.result()

    return samples, result

def train_chess_networks_RL(
    num_iterations=400,
    games_per_iteration=5,
    epsilon_initial=0.3,
    epsilon_final=0.1,
    lr=0.0004,
    gamma=0.95,
    per_move_penalty=1,
    terminal_reward = 200,
    pretrained_model_path_white=None,
    pretrained_model_path_black=None,
    config=None  # Should include any necessary parameters such as config.move_kwargs
):
    # Initialize agents (using your current RL network)
    agent_white = ChessPolicyNet(
        board=chess.Board(),
        color=chess.WHITE,
        device=device,
        epsilon=epsilon_initial
    ).to(device)
    agent_black = ChessPolicyNet(
        board=chess.Board(),
        color=chess.BLACK,
        device=device,
        epsilon=epsilon_initial
    ).to(device)
    
    # Load pretrained weights if provided.
    if pretrained_model_path_white:
        agent_white.load_state_dict(torch.load(pretrained_model_path_white, map_location=device))
    if pretrained_model_path_black:
        agent_black.load_state_dict(torch.load(pretrained_model_path_black, map_location=device))
    
    agent_white.train()
    agent_black.train()

    optimizer_white = torch.optim.Adam(agent_white.parameters(), lr=lr)
    optimizer_black = torch.optim.Adam(agent_black.parameters(), lr=lr)

    # Metrics dictionary to track progress.
    metrics = {
        "white_loss_list": [],
        "black_loss_list": [],
        "white_avg_points": [],
        "black_avg_points": [],
        "game_length_list": [],
        "white_win_rates": [],
        "black_win_rates": [],
        "draw_rates": []
    }

    for iteration in range(num_iterations):
        # Decay epsilon over time.
        epsilon = max(epsilon_final, epsilon_initial - (epsilon_initial - epsilon_final) * (iteration / num_iterations))
        agent_white.epsilon = epsilon
        agent_black.epsilon = epsilon

        print(f"Iteration {iteration+1}/{num_iterations} with epsilon = {epsilon:.4f}")

        game_histories = []
        game_lengths = []
        results = []

        for _ in range(games_per_iteration):
            # Generate one self-play game.
            samples, result = generate_self_play_samples(agent_white, agent_black, config)
            game_lengths.append(len(samples))
            results.append(result)
            print(results[-1])
            
            # (Optionally, you could split samples based on turn or agent.)
            game_histories.append(samples)

        white_game_histories, black_game_histories, white_avg_points, black_avg_points = separate_and_evaluate_game_histories(game_histories, results, terminal_reward, per_move_penalty, agent_white, agent_black)

        white_loss = agent_white.reinforce_update(optimizer_white, white_game_histories, gamma=gamma)
        black_loss = agent_black.reinforce_update(optimizer_black, black_game_histories, gamma=gamma)

        # Compute win/draw statistics.
        white_wins = results.count("1-0")
        black_wins = results.count("0-1")
        draws = results.count("1/2-1/2")
        total_games = len(results)
        metrics["white_win_rates"].append(white_wins / total_games)
        metrics["black_win_rates"].append(black_wins / total_games)
        metrics["draw_rates"].append(draws / total_games)
        metrics["game_length_list"].append(np.mean(game_lengths))
        metrics["white_loss_list"].append(white_loss)
        metrics["black_loss_list"].append(black_loss)
        metrics["white_avg_points"].append(white_avg_points)
        metrics["black_avg_points"].append(black_avg_points)

        # For now, we'll just record loss and win rates.
        print(f"Iteration {iteration+1} completed: Avg Length {np.mean(game_lengths):.2f}, White Loss {white_loss:.4f}, Black Loss {black_loss:.4f}")

    # Optionally, save your metrics to disk.
    save_models_and_metrics(policy_net_white=agent_white, policy_net_black=agent_black, metrics=metrics)

    return metrics

def get_next_run_directory(base_dir="."):
    """Find the next available run directory (run1, run2, ...)."""
    run_number = 1
    while os.path.exists(os.path.join(base_dir, f"run{run_number}")):
        run_number += 1
    return os.path.join(base_dir, f"run{run_number}")

def save_models_and_metrics(policy_net_white, policy_net_black, metrics, base_dir=".", save_to_drive=False):
    """Save models and a training metrics plot in an incrementing run directory."""
    save_dir = get_next_run_directory(base_dir)
    if save_to_drive:
        save_dir = os.path.join('/content/drive/MyDrive/ML_States', save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the models.
    torch.save(policy_net_white.state_dict(), os.path.join(save_dir, "policy_net_white.pth"))
    torch.save(policy_net_black.state_dict(), os.path.join(save_dir, "policy_net_black.pth"))
    print(f"Models saved in {save_dir}")
    
    # Plot the metrics and save the plot.
    fig = plot_training_metrics_binned(metrics)
    plot_path = os.path.join(save_dir, "training_metrics.png")
    fig.savefig(plot_path)
    plt.close(fig)  # Close the figure to free up memory
    print(f"Training metrics plot saved in {plot_path}")

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
    return fig
