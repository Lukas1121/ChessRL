import time
import hf as hf

pretrained_model_path_white = None
pretrained_model_path_black = None

start_time = time.time()

metrics = hf.train_chess_networks_RL(
    num_iterations=12000,
    games_per_iteration=5,
    epsilon_initial=0.3,
    epsilon_final=0.005,
    lr=0.0002,
    terminal_reward=50,
    per_move_penalty = -0.2,
    non_capture_penalty=-0.05, 
    repeat_flip_penalty=-1,
    game_length=120,
    exceed_penalty=-30,
    layers=5,
    pretrained_model_path_white=pretrained_model_path_white,
    pretrained_model_path_black=pretrained_model_path_black,
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total training time: {elapsed_time:.2f} seconds")