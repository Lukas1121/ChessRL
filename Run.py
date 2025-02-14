import Helper_functions as functions



metrics = functions.train_chess_policy_networks(
    num_iterations=2000,
    games_per_iteration=5,
    epsilon_initial=0.1,
    epsilon_final=0.0,
    lr=0.00001,
    gamma=0.95,
    terminal_reward=1000,
    terminal_loss=100,
    per_move_penalty=0,
    non_capture_penalty=-0.15,
    move_length_threshold=120,
    exceeding_length_penalty=0,
    pretrained_model_path_white=None,
    pretrained_model_path_black=None,
)

functions.plot_training_metrics_binned(metrics)