import Helper_functions as hf

pretrained_model_path_white = "run5\policy_net_white.pth"
pretrained_model_path_black = "run5\policy_net_black.pth"

metrics = hf.train_chess_policy_networks(
    num_iterations=1000,
    games_per_iteration=100,
    epsilon_initial=0.4,
    epsilon_final=0.01,
    lr=0.001,
    gamma=0.98,
    terminal_reward=1000,
    terminal_loss=100,
    per_move_penalty=2,
    non_capture_penalty=-0.02,
    move_length_threshold=120,
    method='rl',
    simulations=100,
    pretrained_model_path_white=pretrained_model_path_white,
    pretrained_model_path_black=pretrained_model_path_black,
)

hf.plot_training_metrics_binned(metrics,num_bins=20)
