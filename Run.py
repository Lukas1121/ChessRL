import time
import hf as hf

pretrained_model_path_white = "run18\policy_net_white.pth"
pretrained_model_path_black = "run18\policy_net_black.pth"

start_time = time.time()

metrics = hf.train_chess_networks_RL(
    num_iterations=10000,
    games_per_iteration=5,
    epsilon_initial=0.1,
    epsilon_final=0.05,
    lr=0.0005,
    terminal_reward=1000,
    per_move_penalty = -0.2,
    game_length=120,
    layers=10,
    gamma=0.99,
    pretrained_model_path_white=pretrained_model_path_white,
    pretrained_model_path_black=pretrained_model_path_black,
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total training time: {elapsed_time:.2f} seconds")