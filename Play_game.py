from hf import play_human_vs_bot
import chess

# Provide the paths to your pretrained models for White and Black.
white_model_path = "real_data13\policy_net.pth"
black_model_path = "real_data13\policy_net.pth"

# Optionally, specify your color (e.g., chess.WHITE); if None, colors are assigned at random.
play_human_vs_bot(white_model_path, black_model_path,human_color=chess.BLACK) 

 