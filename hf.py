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
        move, log_prob = agent.choose_move(**config.move_kwargs)
        
        # Optionally, record the network's output (policy prior, etc.) here.
        sample = {
            "state": agent.board_tensor.clone(),
            "target_policy": agent.get_target_policy(),
            "log_prob":agent.log_prob
        }
        samples.append(sample)
        
        if move not in board.legal_moves:
            continue
        
        board.push(move)
        agent.update_board(board)
    
    result = board.result()

    return samples, result

def train_chess_networks_modular(
    num_iterations=400,
    games_per_iteration=5,
    epsilon_initial=0.3,
    epsilon_final=0.1,
    lr=0.0004,
    gamma=0.95,
    pretrained_model_path_white=None,
    pretrained_model_path_black=None,
    config=None  # Should include any necessary parameters such as config.move_kwargs
):
    # Initialize agents (using your current RL network)
    agent_white = ChessPolicyNet(
        board=chess.Board(),
        color=chess.WHITE,
        device=device,
        non_capture_penalty=-1,
        epsilon=epsilon_initial
    ).to(device)
    agent_black = ChessPolicyNet(
        board=chess.Board(),
        color=chess.BLACK,
        device=device,
        non_capture_penalty=-1,
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

        white_samples_all = []
        black_samples_all = []
        game_lengths = []
        results = []

        for _ in range(games_per_iteration):
            # Generate one self-play game.
            samples, result = generate_self_play_samples(agent_white, agent_black, config)
            game_lengths.append(len(samples))
            results.append(result)
            
            # Assign rewards to each sample based on the final game result.
            samples = assign_rewards(samples, result)
            
            # (Optionally, you could split samples based on turn or agent.)
            white_samples_all.extend(samples)
            black_samples_all.extend(samples)

        # Compute win/draw statistics.
        white_wins = results.count("1-0")
        black_wins = results.count("0-1")
        draws = results.count("1/2-1/2")
        total_games = len(results)
        metrics["white_win_rates"].append(white_wins / total_games)
        metrics["black_win_rates"].append(black_wins / total_games)
        metrics["draw_rates"].append(draws / total_games)
        avg_game_length = np.mean(game_lengths)
        metrics["game_length_list"].append(avg_game_length)

        # Now update the networks using the collected self-play samples.
        # Here we use a training function (like your previous train_policy_network) that expects
        # each sample to include: state, target_policy, log_prob, and target_value.
        white_loss = train_policy_network(agent_white, optimizer_white, white_samples_all, gamma=gamma, device=device)
        black_loss = train_policy_network(agent_black, optimizer_black, black_samples_all, gamma=gamma, device=device)

        metrics["white_loss_list"].append(white_loss)
        metrics["black_loss_list"].append(black_loss)

        # Optionally, compute average points (if you're using material scores elsewhere).
        # For now, we'll just record loss and win rates.
        print(f"Iteration {iteration+1} completed: Avg Length {avg_game_length:.2f}, White Loss {white_loss:.4f}, Black Loss {black_loss:.4f}")

        # Save model checkpoints periodically.
        if save_to_drive and ((iteration + 1) % 50 == 0):
            torch.save(agent_white.state_dict(), f"agent_white_{iteration+1}.pth")
            torch.save(agent_black.state_dict(), f"agent_black_{iteration+1}.pth")

    # Optionally, save your metrics to disk.
    save_models_and_metrics(policy_net_white=agent_white, policy_net_black=agent_black, save_to_drive=save_to_drive, metrics=metrics)

    return metrics
