from Helper_functions import play_human_vs_bot
import chess

# Provide the paths to your pretrained models for White and Black.
white_model_path = "run2/policy_net_white.pth"
black_model_path = "run2/policy_net_black.pth"

# Optionally, specify your color (e.g., chess.WHITE); if None, colors are assigned at random.
play_human_vs_bot(white_model_path, black_model_path, method='rl',minmax_depth=4,top_n=2) 

 