import time
import hf as hf

pretrained_model_path_white = "run13\policy_net_white.pth"
pretrained_model_path_black = "run13\policy_net_black.pth"

start_time = time.time()

metrics = hf.train_chess_networks_RL(
    num_iterations=2000,
    games_per_iteration=10,
    epsilon_initial=0.3,
    epsilon_final=0.005,
    lr=0.0001,
    terminal_reward=50,
    per_move_penalty = -0.2,
    non_capture_penalty=-0.15, 
    repeat_flip_penalty=-1,
    game_length=120,
    exceed_penalty=-50,
    layers=5,
    pretrained_model_path_white=pretrained_model_path_white,
    pretrained_model_path_black=pretrained_model_path_black,
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total training time: {elapsed_time:.2f} seconds")