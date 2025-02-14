import Helper_functions as hf

metrics = hf.train_chess_policy_networks(
    num_iterations=1000,
    games_per_iteration=10,
    epsilon_initial=0.1,
    epsilon_final=0.0,
    lr=0.00001,
    gamma=0.95,
    terminal_reward=1000,
    terminal_loss=100,
    per_move_penalty=1,
    non_capture_penalty=-0.15,
    move_length_threshold=120,
    exceeding_length_penalty=100,
    pretrained_model_path_white="run0/policy_net_white.pth",
    pretrained_model_path_black="run0/policy_net_black.pth",)

hf.plot_training_metrics_binned(metrics)