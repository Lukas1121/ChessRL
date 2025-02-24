import time
import hf as hf

pretrained_model_path_white = None
pretrained_model_path_black = None

start_time = time.time()

metrics = hf.train_chess_networks_RL(
    num_iterations=2000,
    games_per_iteration=20,
    epsilon_initial=0.5,
    epsilon_final=0.1,
    lr=0.001,
    terminal_reward=200,
    per_move_penalty = 1,
    gamma=0.98,
    pretrained_model_path_white=pretrained_model_path_white,
    pretrained_model_path_black=pretrained_model_path_black,
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total training time: {elapsed_time:.2f} seconds")