import Helper_functions as hf

pretrained_model_path_white = "run1\policy_net_white.pth"
pretrained_model_path_black = "run1\policy_net_black.pth"

metrics = hf.train_chess_policy_networks(
    num_iterations=50,
    games_per_iteration=1,
    epsilon_initial=0.0,
    epsilon_final=0.0,
    lr=0.005,
    gamma=0.95,
    terminal_reward=1000,
    terminal_loss=100,
    per_move_penalty=0.5,
    non_capture_penalty=-0.05,
    move_length_threshold=120,
    exceeding_length_penalty=100,
    use_lookahead=True,
    minmax_depth=2,
    top_n=5,
    pretrained_model_path_white=pretrained_model_path_white,
    pretrained_model_path_black=pretrained_model_path_black,
)

hf.plot_training_metrics_binned(metrics,num_bins=5)