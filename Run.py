import Helper_functions as hf

pretrained_model_path_white = "policy_net_white.pth"
pretrained_model_path_black = "policy_net_black.pth"

metrics = hf.train_chess_policy_networks(
    num_iterations=1000,
    games_per_iteration=100,
    epsilon_initial=0.3,
    epsilon_final=0.0,
    lr=0.001,
    gamma=0.95,
    terminal_reward=1000,
    terminal_loss=100,
    per_move_penalty=1,
    non_capture_penalty=-0.05,
    move_length_threshold=120,
    exceeding_length_penalty=100,
    pretrained_model_path_white=pretrained_model_path_white,
    pretrained_model_path_black=pretrained_model_path_black,
)

hf.plot_training_metrics_binned(metrics)