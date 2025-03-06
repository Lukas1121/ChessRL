# Hf.py

import os
import chess
import chess.pgn
from ChessRL import ChessPolicyNet,ChessPositionalEvaluator
import random
import os
import seaborn as sns
import numpy as np
from itertools import cycle
import torch
import torch.nn.functional as F
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use the first available GPU
    print("GPU is available and being used.")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU instead.")
import matplotlib.pyplot as plt

def separate_and_evaluate_game_histories(agent_white, agent_black, game_histories, results, terminal_reward, per_move_penalty,
                                           max_game_length, exceed_penalty, non_capture_penalty=0, repeat_flip_penalty=0):
    """
    Given a list of game histories (each a list of move sample dicts), the game results,
    and a terminal reward, compute the reward for each move and separate the moves into
    white and black game histories.
    
    Each move sample is expected to contain:
      - 'state': a torch.Tensor of shape [12, 8, 8]
      - 'policy_info': the output from move selection (e.g. log_prob)
      - 'move': (optional) a chess.Move object representing the move
      - 'board': (optional) the chess.Board object from which the move was made,
          OR an 'is_capture' boolean flag can be provided.
    
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
    
    New Parameters:
      - non_capture_penalty: extra penalty subtracted if the move is not a capture.
      - repeat_flip_penalty: extra penalty subtracted if the current move is exactly the reverse of
                             the immediately previous move.
    
    Returns:
      - white_game_histories: List of game histories (one per game) for white moves.
      - black_game_histories: List of game histories (one per game) for black moves.
      - avg_white_points: The average final score for white across the batch.
      - avg_black_points: The average final score for black across the batch.
    """
    white_game_histories = []
    black_game_histories = []
    white_final_scores = []
    black_final_scores = []
    
    # Process each game.
    for game_index, game in enumerate(game_histories):
        white_moves = []
        black_moves = []
        
        # Process each move sample in the game.
        for i, sample in enumerate(game):
            move = sample.get('move', None)
            # Determine if the move is a capture.
            is_capture = False
            if move is not None:
                if 'board' in sample:
                    is_capture = sample['board'].is_capture(move)
                else:
                    is_capture = sample.get('is_capture', False)
            
            # Compute the intermediate reward.
            if i % 2 == 0:  # White's move.
                reward = agent_white.evaluator(sample['state'], chess.WHITE).item()
                sample['reward'] = reward
                if not is_capture:
                    sample['reward'] = non_capture_penalty
                # Check if the current move reverses the immediate previous move.
                if i > 0:
                    prev_move = game[i-1].get('move', None)
                    if prev_move is not None and move is not None:
                        reverse_prev = type(prev_move)(prev_move.to_square, prev_move.from_square)
                        if move == reverse_prev:
                            sample['reward'] = repeat_flip_penalty
                white_moves.append(sample)
            else:  # Black's move.
                reward = -agent_black.evaluator(sample['state'], chess.BLACK).item()
                sample['reward'] = reward
                if not is_capture:
                    sample['reward'] = non_capture_penalty
                if i > 0:
                    prev_move = game[i-1].get('move', None)
                    if prev_move is not None and move is not None:
                        reverse_prev = type(prev_move)(prev_move.to_square, prev_move.from_square)
                        if move == reverse_prev:
                            sample['reward'] = repeat_flip_penalty
                black_moves.append(sample)
        
        # Adjust final move rewards based on the game result.
        result = results[game_index]
        if result == "1-0":
            if white_moves:
                white_moves[-1]['reward'] += terminal_reward
            if black_moves:
                black_moves[-1]['reward'] -= terminal_reward/2
        elif result == "0-1":
            if white_moves:
                white_moves[-1]['reward'] -= terminal_reward/2
            if black_moves:
                black_moves[-1]['reward'] += terminal_reward
        else:  # Draw ("1/2-1/2")
            penalty = per_move_penalty * (len(game) / 2)
            white_moves[-1]['reward'] += penalty
            black_moves[-1]['reward'] += penalty

        # Apply penalty if game length exceeds maximum.
        if len(game) >= max_game_length:
            if white_moves:
                white_moves[-1]['reward'] += exceed_penalty
            if black_moves:
                black_moves[-1]['reward'] += exceed_penalty

        white_game_histories.append(white_moves)
        black_game_histories.append(black_moves)
        white_final_scores.append(white_moves[-1]['reward'])
        black_final_scores.append(black_moves[-1]['reward'])
    
    white_avg_points = np.mean(white_final_scores) if white_final_scores else 0
    black_avg_points = np.mean(black_final_scores) if black_final_scores else 0

    return white_game_histories, black_game_histories, white_avg_points, black_avg_points


def generate_self_play_samples(agent_white, agent_black, game_length, config):
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
        if len(samples) >= 2*game_length:
            break
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
            "move" : move
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
    game_length=120,
    epsilon_initial=0.3,
    epsilon_final=0.1,
    lr=0.0004,
    evaluator_lr=0.00005,  # Slower learning rate for the evaluator
    gamma=0.95,
    non_capture_penalty=0, 
    repeat_flip_penalty=0,
    per_move_penalty=-1,
    exceed_penalty=-50,
    layers=3,
    terminal_reward=200,
    pretrained_model_path_white=None,
    pretrained_model_path_black=None,
    pretrained_evaluator_path=None,  # Path to a pretrained evaluator
    config=None
):
    # Create a shared positional evaluator
    evaluator = ChessPositionalEvaluator(device=device)
    
    # Load pretrained evaluator if provided
    if pretrained_evaluator_path:
        evaluator.load_state_dict(torch.load(pretrained_evaluator_path, map_location=device))
    
    # Initialize agents with the shared evaluator
    agent_white = ChessHybridNet(
        board=chess.Board(),
        color=chess.WHITE,
        device=device,
        layers=layers,
        epsilon=epsilon_initial,
        evaluator=evaluator
    ).to(device)
    
    agent_black = ChessHybridNet(
        board=chess.Board(),
        color=chess.BLACK,
        device=device,
        layers=layers,
        epsilon=epsilon_initial,
        evaluator=evaluator
    ).to(device)
    
    # Load pretrained weights if provided
    if pretrained_model_path_white:
        # Filter out evaluator parameters when loading
        pretrained_dict = torch.load(pretrained_model_path_white, map_location=device)
        model_dict = agent_white.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'evaluator' not in k}
        model_dict.update(pretrained_dict)
        agent_white.load_state_dict(model_dict)
        
    if pretrained_model_path_black:
        # Filter out evaluator parameters when loading
        pretrained_dict = torch.load(pretrained_model_path_black, map_location=device)
        model_dict = agent_black.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'evaluator' not in k}
        model_dict.update(pretrained_dict)
        agent_black.load_state_dict(model_dict)
    
    agent_white.train()
    agent_black.train()

    # Create optimizers with different learning rates for policy and evaluator
    optimizer_white = torch.optim.Adam([
        {'params': [p for n, p in agent_white.named_parameters() if 'evaluator' not in n], 'lr': lr},
        {'params': evaluator.parameters(), 'lr': evaluator_lr}  # Slower learning for tables
    ])
    
    optimizer_black = torch.optim.Adam([
        {'params': [p for n, p in agent_black.named_parameters() if 'evaluator' not in n], 'lr': lr}
        # Evaluator already included in white's optimizer
    ])

    # Metrics dictionary to track progress
    metrics = {
        "white_loss_list": [],
        "black_loss_list": [],
        "white_avg_points": [],
        "black_avg_points": [],
        "game_length_list": [],
        "white_win_rates": [],
        "black_win_rates": [],
        "draw_rates": [],
        "evaluator_tables": []  # Track table changes
    }

    for iteration in range(num_iterations):
        # Decay epsilon over time
        epsilon = max(epsilon_final, epsilon_initial - (epsilon_initial - epsilon_final) * (iteration / num_iterations))
        agent_white.epsilon = epsilon
        agent_black.epsilon = epsilon

        print(f"Iteration {iteration+1}/{num_iterations} with epsilon = {epsilon:.4f}")

        game_histories = []
        game_lengths = []
        results = []

        for _ in range(games_per_iteration):
            # Generate one self-play game
            samples, result = generate_self_play_samples(agent_white, agent_black, game_length, config)
            game_lengths.append(len(samples))
            results.append(result)
            print(results[-1])
            
            game_histories.append(samples)

        # Modified to use evaluator directly in the reward calculation
        (white_game_histories, 
         black_game_histories, 
         white_avg_points, 
         black_avg_points) = separate_and_evaluate_game_histories(
             agent_white, agent_black, game_histories, results, 
             terminal_reward, per_move_penalty, game_length, exceed_penalty, 
             non_capture_penalty, repeat_flip_penalty)
        
        # Updated reinforce_update methods that now train the evaluator too
        white_loss = agent_white.reinforce_update(optimizer_white, white_game_histories, gamma=gamma)
        black_loss = agent_black.reinforce_update(optimizer_black, black_game_histories, gamma=gamma)

        # Compute win/draw statistics
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
        
        # Periodically save table snapshots to track evolution
        if iteration % 10 == 0:
            # Save a copy of the current evaluator state (tables only)
            table_snapshot = {name: param.clone().detach().cpu() 
                             for name, param in evaluator.named_parameters() 
                             if 'table' in name}
            metrics["evaluator_tables"].append((iteration, table_snapshot))

        # Log the progress
        print(f"Iteration {iteration+1} completed: Avg Length {np.mean(game_lengths)/2:.2f}, "
              f"White win rate {white_wins / total_games:.2f}, Black win rate {black_wins / total_games:.2f}, "
              f"Draw rate {draws / total_games:.2f}")

        # Save checkpoint periodically
        if (iteration + 1) % 100 == 0:
            print(f"Saving checkpoint at iteration {iteration+1}...")
            save_models_and_metrics(
                policy_net_white=agent_white,
                policy_net_black=agent_black,
                evaluator=evaluator,  # Also save the evaluator separately
                metrics=metrics,
                iteration=iteration+1
            )
            
            # # Visualize how tables are evolving
            # if hasattr(evaluator, 'pawn_table_middlegame'):
            #     visualize_tables_evolution(evaluator, iteration+1)
            
    # Save final models and metrics
    save_models_and_metrics(
        policy_net_white=agent_white,
        policy_net_black=agent_black,
        evaluator=evaluator,
        metrics=metrics,
        iteration=num_iterations
    )

    return metrics, evaluator

def get_next_run_directory(base_dir=".", folder_name="run"):
    """Find the next available run directory (run1, run2, ...)."""
    run_number = 1
    while os.path.exists(os.path.join(base_dir, f"{folder_name}{run_number}")):
        run_number += 1
    return os.path.join(base_dir, f"{folder_name}{run_number}")

def save_models_and_metrics(policy_net_white, policy_net_black, metrics, evaluator=None, 
                            base_dir=".", folder_name="run", iteration=None):
    """Save models, evaluator, and training metrics in an incrementing run directory."""
    # If iteration is provided, use it as a subfolder
    if iteration is not None:
        subfolder = f"iter_{iteration}"
    else:
        subfolder = ""
    
    # Get the base save directory
    if not os.path.exists(os.path.join(base_dir, folder_name)):
        save_dir = os.path.join(base_dir, folder_name)
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = os.path.join(base_dir, folder_name)
    
    # Add subfolder if needed
    if subfolder:
        save_dir = os.path.join(save_dir, subfolder)
        os.makedirs(save_dir, exist_ok=True)
    
    # Save the models
    torch.save(policy_net_white.state_dict(), os.path.join(save_dir, "policy_net_white.pth"))
    torch.save(policy_net_black.state_dict(), os.path.join(save_dir, "policy_net_black.pth"))
    
    # Save the evaluator if provided
    if evaluator is not None:
        torch.save(evaluator.state_dict(), os.path.join(save_dir, "positional_evaluator.pth"))
        
        # Create a tables directory to store visualizations
        tables_dir = os.path.join(save_dir, "tables")
        os.makedirs(tables_dir, exist_ok=True)
        
        # Visualize all piece tables for each phase
        for piece in ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']:
            for phase in ['opening', 'middlegame', 'endgame']:
                attr_name = f"{piece}_table_{phase}"
                if hasattr(evaluator, attr_name):
                    table = getattr(evaluator, attr_name)
                    
                    plt.figure(figsize=(6, 6))
                    table_np = table.detach().cpu().numpy()
                    sns.heatmap(table_np, annot=True, cmap='coolwarm', 
                              vmin=-0.3, vmax=0.3, fmt='.2f')
                    plt.title(f"{piece.capitalize()} {phase}")
                    plt.savefig(os.path.join(tables_dir, f"{piece}_{phase}.png"))
                    plt.close()
    
    print(f"Models saved in {save_dir}")
    
    # Save and plot metrics if provided
    if metrics is not None:
        # Save raw metrics data
        metrics_path = os.path.join(save_dir, "training_metrics.pkl")
        with open(metrics_path, 'wb') as f:
            pickle.dump({k: v for k, v in metrics.items() if k != "evaluator_tables"}, f)
        
        # If we're tracking table evolution, save that separately
        if "evaluator_tables" in metrics:
            tables_evolution_path = os.path.join(save_dir, "table_evolution.pt")
            torch.save(metrics["evaluator_tables"], tables_evolution_path)
        
        # Create and save the metrics plot
        fig = plot_training_metrics_binned(metrics)
        plot_path = os.path.join(save_dir, "training_metrics.png")
        fig.savefig(plot_path)
        plt.close(fig)  # Close the figure to free up memory
        
        print(f"Training metrics plot saved in {plot_path}")
    else:
        print(f"Models saved to {save_dir}")

def plot_training_metrics_binned(metrics, num_bins=20):
    """Plot training metrics by binning the iteration data to provide a clearer view of trends."""
    # Set up a 2x3 grid for plotting
    fig, axs = plt.subplots(2, 2, figsize=(18, 10))

    # 1. Binned REINFORCE Loss for White and Black
    iterations, white_loss_means, white_loss_stds = bin_data(metrics['white_loss_list'], num_bins)
    _, black_loss_means, black_loss_stds = bin_data(metrics['black_loss_list'], num_bins)
    axs[0, 0].errorbar(iterations, white_loss_means, yerr=white_loss_stds, marker='o', color='blue', label='White Loss')
    axs[0, 0].errorbar(iterations, black_loss_means, yerr=black_loss_stds, marker='o', color='red', label='Black Loss')
    axs[0, 0].set_xlabel("Iteration")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].set_title("Binned REINFORCE Loss per Iteration")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. Binned Average Points per Iteration
    iterations, white_avg_points_means, white_avg_points_stds = bin_data(metrics['white_avg_points'], num_bins)
    _, black_avg_points_means, black_avg_points_stds = bin_data(metrics['black_avg_points'], num_bins)
    axs[0, 1].errorbar(iterations, white_avg_points_means, yerr=white_avg_points_stds, marker='o', color='blue', label='White Avg Points')
    axs[0, 1].errorbar(iterations, black_avg_points_means, yerr=black_avg_points_stds, marker='o', color='red', label='Black Avg Points')
    axs[0, 1].set_xlabel("Iteration")
    axs[0, 1].set_ylabel("Avg Points")
    axs[0, 1].set_title("Binned Average Points per Iteration")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 3. Binned Average Game Length per Iteration
    iterations, game_length_means, game_length_stds = bin_data(metrics['game_length_list'], num_bins)
    axs[1, 1].errorbar(iterations, game_length_means, yerr=game_length_stds, marker='o', color='purple')
    axs[1, 1].set_xlabel("Iteration")
    axs[1, 1].set_ylabel("Moves")
    axs[1, 1].set_title("Binned Average Game Length per Iteration")
    axs[1, 1].grid(True)

    # 4. Binned Win/Loss/Draw Rates per Iteration
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

def visualize_table_evolution(metrics, piece="pawn", phase="middlegame", num_iterations=5):
    """Visualize how a specific piece-square table evolved during training."""
    if "evaluator_tables" not in metrics:
        print("No table evolution data found in metrics.")
        return
    
    # Extract the table evolution data
    table_data = metrics["evaluator_tables"]
    
    # Select iterations to visualize (evenly spaced)
    if len(table_data) <= num_iterations:
        iterations_to_show = range(len(table_data))
    else:
        step = len(table_data) // num_iterations
        iterations_to_show = range(0, len(table_data), step)
    
    # Create a grid of subplots
    fig, axs = plt.subplots(1, len(iterations_to_show), figsize=(15, 5))
    if len(iterations_to_show) == 1:
        axs = [axs]  # Make it iterable if only one subplot
    
    # Plot each selected iteration
    for i, idx in enumerate(iterations_to_show):
        iteration, tables = table_data[idx]
        table_key = f"{piece}_table_{phase}"
        
        if table_key in tables:
            table = tables[table_key]
            sns.heatmap(table.numpy(), annot=True, cmap='coolwarm', 
                       vmin=-0.3, vmax=0.3, ax=axs[i], fmt='.2f')
            axs[i].set_title(f"Iteration {iteration}")
            
            # Remove tick labels except for the first plot
            if i > 0:
                axs[i].set_ylabel('')
                axs[i].set_yticklabels([])
    
    fig.suptitle(f"Evolution of {piece.capitalize()} {phase} table", fontsize=16)
    plt.tight_layout()
    return fig

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

def train_chess_networks_hybrid(
    num_iterations=400,
    games_per_iteration=5,
    game_length = 120,
    epsilon_initial=0.3,
    epsilon_final=0.1,
    lr=0.0004,
    non_capture_penalty=0, 
    repeat_flip_penalty=0,
    per_move_penalty=-1,
    exceed_penalty=-50,
    layers=10,
    terminal_reward = 200,
    simulations = 10,
    pretrained_model_path_white=None,
    pretrained_model_path_black=None,
    config=None  # Should include any necessary parameters such as config.move_kwargs
):
    # Initialize agents using the hybrid network.
    agent_white = ChessHybridNet(
        board=chess.Board(),
        color=chess.WHITE,
        device=device,
        layers=layers
    ).to(device)
    agent_black = ChessHybridNet(
        board=chess.Board(),
        color=chess.BLACK,
        device=device,
        layers=layers
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
        # Decay epsilon over time (if your hybrid agent uses it in choose_move for any exploration).
        epsilon = max(epsilon_final, epsilon_initial - (epsilon_initial - epsilon_final) * (iteration / num_iterations))
        agent_white.epsilon = epsilon
        agent_black.epsilon = epsilon

        print(f"Iteration {iteration+1}/{num_iterations} with epsilon = {epsilon:.4f}")

        game_histories = []
        game_lengths = []
        results = []

        for _ in range(games_per_iteration):
            # Generate one self-play game.
            samples, result = generate_self_play_samples(agent_white, agent_black, game_length, config)
            game_lengths.append(len(samples))
            results.append(result)
            print(f"Game result: {result}")
            
            # (Optionally, you could split samples based on turn or agent.)
            game_histories.append(samples)

        (white_game_histories, 
         black_game_histories, 
         white_avg_points, 
         black_avg_points) = separate_and_evaluate_game_histories(agent_white, agent_black, game_histories, results, terminal_reward, per_move_penalty,
                                           game_length, exceed_penalty, non_capture_penalty, repeat_flip_penalty)

        # Perform policy updates using the agents’ reinforcement learning update routines.
        white_loss = agent_white.reinforce_update(optimizer_white, white_game_histories)
        black_loss = agent_black.reinforce_update(optimizer_black, black_game_histories)

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

        # Log the progress.
        print(f"Iteration {iteration+1} completed: Avg Length {np.mean(game_lengths):.2f}, "
              f"White win rate {white_wins / total_games:.4f}, Black win rate {black_wins / total_games:.4f}, "
              f"Draw rate {draws / total_games:.4f}")

        # Save checkpoint every 100 iterations.
        if (iteration + 1) % 100 == 0:
            print(f"Saving checkpoint at iteration {iteration+1}...")
            save_models_and_metrics(
                policy_net_white=agent_white,
                policy_net_black=agent_black,
                metrics=metrics
            )

    # Optionally, save your final metrics to disk.
    save_models_and_metrics(policy_net_white=agent_white, policy_net_black=agent_black, metrics=metrics)

    return metrics

def count_conv_layers(model_path: str) -> int:
    """
    Counts the number of convolutional layers in a PyTorch .pth file by inspecting the state dictionary.
    
    This function assumes that the keys for the convolutional layers in the state dictionary
    contain the substring 'conv_layers' and that each convolutional layer has a weight parameter
    (e.g., 'conv_layers.0.weight').
    
    Parameters:
        model_path (str): Path to the .pth checkpoint file.
        
    Returns:
        int: The number of convolutional layers found in the checkpoint.
    """
    state_dict = torch.load(model_path, map_location=device)
    # Count keys corresponding to convolutional layer weights.
    conv_layer_weight_keys = [
        key for key in state_dict.keys() 
        if key.startswith("conv_layers") and key.endswith("weight")
    ]
    return len(conv_layer_weight_keys)

def play_human_vs_bot(white_model_path, black_model_path, human_color=None):
    """
    Play a game of chess between a human and the bot.
    
    Parameters:
        white_model_path (str): Path to the pretrained model file for White.
        black_model_path (str): Path to the pretrained model file for Black.
        human_color (chess.WHITE or chess.BLACK or None): The color you wish to play.
            If set to None, the bot's color is chosen at random, and you play the opposite.
        method (str): The method for move selection ('rl', 'mcts', 'lookahead').
        minmax_depth (int): The depth to search when using minimax.
        top_n (int): Number of candidate moves to evaluate in minimax.
        simulations (int): Number of simulations for MCTS.
    
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
    layers = count_conv_layers(white_model_path)
    
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
            board=board,
            layers=layers,
            device=device,
            color=chess.WHITE
        ).to(device)
        policy_net.load_state_dict(torch.load(white_model_path, map_location=device))
    else:
        policy_net = ChessPolicyNet(
            board=board,
            layers=layers,
            device=device,
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
            move, _ = policy_net.choose_move()
            
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
