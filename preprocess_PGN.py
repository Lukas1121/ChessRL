import os
import torch
import chess
import chess.pgn
import numpy as np
from typing import List, Tuple

class ChessDataset:
    def __init__(self, output_dir: str, checkpoint_interval: int = 20000):
        """
        Efficient chess game data preprocessor and storage.
        
        Uses compressed numpy arrays and more memory-efficient storage.
        
        Args:
            output_dir (str): Directory to save preprocessed data
            checkpoint_interval (int): Number of games per checkpoint
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.checkpoint_interval = checkpoint_interval
        self.current_checkpoint = 1
        
        # Efficient storage containers
        self.state_tensors = []
        self.move_indices = []
        self.game_results = []
        self.move_ucis = []

    def board_to_compact_array(self, board: chess.Board) -> np.ndarray:
        """
        Convert board to a more compact numpy representation.
        
        Uses uint8 dtype and a more compact encoding.
        
        Returns:
            np.ndarray: Compact board representation (12, 8, 8) with uint8
        """
        compact_board = np.zeros((12, 8, 8), dtype=np.uint8)
        for square, piece in board.piece_map().items():
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            # Compact channel mapping
            channel = piece.piece_type - 1 + (0 if piece.color == chess.WHITE else 6)
            compact_board[channel, row, col] = 1
        return compact_board

    def process_pgn(
        self, 
        pgn_file_path: str, 
        max_games: int = None, 
        verbose: bool = True
    ) -> int:
        """
        Process PGN file with efficient memory management.
        
        Args:
            pgn_file_path (str): Path to PGN file
            max_games (int, optional): Maximum games to process
            verbose (bool): Print processing details
        
        Returns:
            int: Total games processed
        """
        # Create action space mapping
        from ChessRL import create_action_space
        action_space = create_action_space()
        move_to_idx = {move.uci(): idx for idx, move in enumerate(action_space)}
        
        total_games = 0
        with open(pgn_file_path, "r", encoding="utf-8") as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                
                result = game.headers.get("Result", "Unknown")
                board = game.board()
                
                for move in game.mainline_moves():
                    move_uci = move.uci()
                    
                    # Check if move is in action space
                    if move_uci in move_to_idx:
                        state_array = self.board_to_compact_array(board)
                        move_index = move_to_idx[move_uci]
                        
                        # Store data
                        self.state_tensors.append(state_array)
                        self.move_indices.append(move_index)
                        self.move_ucis.append(move_uci)
                        self.game_results.append(result)
                    
                    board.push(move)
                
                total_games += 1
                
                # Checkpoint mechanism
                if total_games % self.checkpoint_interval == 0:
                    self._save_checkpoint()
                
                if max_games and total_games >= max_games:
                    break
        
        # Save any remaining data
        if self.state_tensors:
            self._save_checkpoint(is_final=True)
        
        if verbose:
            print(f"Processed {total_games} games")
        
        return total_games

    def _save_checkpoint(self, is_final: bool = False):
        """
        Save current data checkpoint with efficient compression.
        
        Uses numpy's savez_compressed for space-efficient storage.
        """
        if not self.state_tensors:
            return
        
        # Convert lists to numpy arrays
        state_array = np.array(self.state_tensors)
        move_indices_array = np.array(self.move_indices)
        results_array = np.array(self.game_results)
        move_ucis_array = np.array(self.move_ucis)
        
        # Create checkpoint filename
        checkpoint_name = f"checkpoint_{self.current_checkpoint:04d}"
        if is_final:
            checkpoint_name += "_final"
        checkpoint_path = os.path.join(self.output_dir, checkpoint_name + ".npz")
        
        # Save compressed numpy arrays
        np.savez_compressed(
            checkpoint_path, 
            states=state_array, 
            moves=move_indices_array, 
            results=results_array, 
            ucis=move_ucis_array
        )
        
        print(f"Saved checkpoint: {checkpoint_path}")
        print(f"Checkpoint data sizes:")
        print(f"  States: {state_array.nbytes / (1024*1024):.2f} MB")
        print(f"  Move Indices: {move_indices_array.nbytes / (1024*1024):.2f} MB")
        print(f"  Results: {results_array.nbytes / (1024*1024):.2f} MB")
        
        # Reset data containers
        self.state_tensors = []
        self.move_indices = []
        self.game_results = []
        self.move_ucis = []
        
        self.current_checkpoint += 1

    def load_checkpoint(self, checkpoint_file: str) -> dict:
        """
        Load a previously saved checkpoint.
        
        Returns:
            dict: Checkpoint data with states, moves, results, and UCIs
        """
        checkpoint = np.load(checkpoint_file)
        return {
            'states': torch.from_numpy(checkpoint['states']),
            'moves': torch.from_numpy(checkpoint['moves']),
            'results': checkpoint['results'],
            'ucis': checkpoint['ucis']
        }

# Example usage
if __name__ == "__main__":
    pgn_file = "data/LumbrasGigaBase 2024.pgn"
    output_directory = "pgn_checkpoints_compressed"
    
    processor = ChessDataset(output_directory)
    processed_games = processor.process_pgn(
        pgn_file_path=pgn_file,
        max_games=10000000,  # Adjust as needed
        verbose=True
    )