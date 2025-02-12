from Helper_functions import play_human_vs_bot

# Provide the paths to your pretrained models for White and Black.
white_model_path = "path/to/your/white_model.pth"
black_model_path = "path/to/your/black_model.pth"

# Optionally, specify your color (e.g., chess.WHITE); if None, colors are assigned at random.
play_human_vs_bot(white_model_path, black_model_path, human_color=None)
