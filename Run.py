import time
import hf as hf

pretrained_model_path_white = "run1\policy_net_white.pth"
pretrained_model_path_black = "run1\policy_net_black.pth"

start_time = time.time()

metrics = hf.train_chess_networks_RL(
    num_iterations=12000,
    games_per_iteration=4,
    epsilon_initial=0.5,
    epsilon_final=0.05,
    lr=0.0005,
    terminal_reward=250,
    per_move_penalty = -0.1,
    non_capture_penalty=-0.1, 
    repeat_flip_penalty=-1,
    game_length=120,
    exceed_penalty=-50,
    layers=10,
    gamma=0.98,
    pretrained_model_path_white=pretrained_model_path_white,
    pretrained_model_path_black=pretrained_model_path_black,
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total training time: {elapsed_time:.2f} seconds")