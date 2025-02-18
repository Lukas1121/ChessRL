import Helper_functions as hf

pretrained_model_path_white = None
pretrained_model_path_black = None

metrics = hf.train_chess_policy_networks(
    num_iterations=20,
    games_per_iteration=1,
    epsilon_initial=0.5,
    epsilon_final=0.5,
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
