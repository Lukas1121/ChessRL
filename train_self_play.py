import time
import hf as hf

pretrained_model_path_white = "simplified_real_data17\policy_net.pth"
pretrained_model_path_black = "simplified_real_data17\policy_net.pth"

pretrained_evaluator_path = "run\iter_50\positional_evaluator.pth"

start_time = time.time()

metrics, evaluator = hf.train_chess_networks_RL(
    num_iterations=500,
    games_per_iteration=5,
    epsilon_initial=0.05,
    epsilon_final=0.0,
    lr=0.0005,
    evaluator_lr=0.0005,
    terminal_reward=50,
    per_move_penalty = 0.05,
    non_capture_penalty=-0.05, 
    repeat_flip_penalty=-0.2,
    game_length=120,
    exceed_penalty=-10,
    layers=12,
    pretrained_model_path_white=pretrained_model_path_white,
    pretrained_model_path_black=pretrained_model_path_black,
    pretrained_evaluator_path=pretrained_evaluator_path,
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total training time: {elapsed_time:.2f} seconds")