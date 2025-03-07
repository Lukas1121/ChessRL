import os
import chess
from ChessRL import ChessPolicyNet
import random
from itertools import cycle
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import glob
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")  # Use the first available GPU
    print("GPU is available and being used.")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU instead.")

def get_next_run_directory(base_dir=".", folder_name="run"):
    """Find the next available run directory (run1, run2, ...)."""
    run_number = 1
    while os.path.exists(os.path.join(base_dir, f"{folder_name}{run_number}")):
        run_number += 1
    return os.path.join(base_dir, f"{folder_name}{run_number}")

def save_models_and_metrics(policy_net, metrics, base_dir=".", save_to_drive=False, folder_name="real_data"):
    """
    Save the model and a training metrics plot in an incrementing run directory.
    
    This version is modified for supervised training on preprocessed data.
    It saves the state_dict of the single model and, if metrics is not None, creates a plot
    of binned training loss.
    """
    save_dir = get_next_run_directory(base_dir, folder_name)
    if save_to_drive:
        save_dir = os.path.join('/content/drive/MyDrive/ML_States', save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the model.
    torch.save(policy_net.state_dict(), os.path.join(save_dir, "policy_net.pth"))
    print(f"Model saved in {save_dir}")
    
    fig = plot_training_metrics_binned(metrics)
    plot_path = os.path.join(save_dir, "training_metrics.png")
    fig.savefig(plot_path)
    plt.close(fig)  # Free up memory by closing the figure.
    print(f"Training metrics plot saved in {plot_path}")

def bin_data(data, num_bins):
    """
    Splits a list/array of data into a specified number of bins,
    and computes the mean and standard deviation in each bin.
    """
    data = np.array(data)
    n = len(data)
    bin_size = int(np.ceil(n / num_bins))
    bin_means = []
    bin_stds = []
    bin_centers = []
    for i in range(0, n, bin_size):
        bin_slice = data[i:i+bin_size]
        bin_means.append(np.mean(bin_slice))
        bin_stds.append(np.std(bin_slice))
        bin_centers.append(i + len(bin_slice) / 2.0)
    return np.array(bin_centers), np.array(bin_means), np.array(bin_stds)

def plot_training_metrics_binned(metrics, num_bins=20):
    """
    Plot training metrics by binning the iteration data.
    """
    losses = metrics.get("losses", [])
    if len(losses) == 0:
        print("No loss data to plot.")
        fig, ax = plt.subplots()
        return fig
    
    # Create iteration numbers corresponding to each loss.
    iterations = np.arange(1, len(losses) + 1)
    
    # Bin the loss data.
    bin_centers, bin_means, bin_stds = bin_data(losses, num_bins)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(bin_centers, bin_means, yerr=bin_stds, marker='o', linestyle='-', color='blue')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Binned Training Loss per Iteration")
    ax.grid(True)
    
    return fig

# The simplified policy network that doesn't use the evaluator
class SimplePolicyNet(torch.nn.Module):
    def __init__(self, layers=5):
        super(SimplePolicyNet, self).__init__()
        
        conv_layers = []
        # First layer: convert input channels (12) to 128
        conv_layers.append(torch.nn.Conv2d(in_channels=12, out_channels=128, kernel_size=3, padding=1))
        conv_layers.append(torch.nn.ReLU())
        
        if layers > 2:
            # Add (layers - 2) intermediate layers maintaining 128 channels
            for _ in range(layers - 2):
                conv_layers.append(torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1))
                conv_layers.append(torch.nn.ReLU())
        
        # Keep the last layer at 128 channels as well
        if layers > 1:
            conv_layers.append(torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1))
            conv_layers.append(torch.nn.ReLU())
        
        self.conv_layers = torch.nn.Sequential(*conv_layers)
        
        # After the conv layers, the spatial dimensions remain 8x8
        self.fc1 = torch.nn.Linear(128 * 8 * 8, 512)
        self.relu = torch.nn.ReLU()
        
        from ChessRL import create_action_space
        action_space = create_action_space()
        num_actions = len(action_space)
        self.policy_head = torch.nn.Linear(512, num_actions)
    
    def forward(self, x):
        # Process through convolutional layers
        x = self.conv_layers(x)
        
        # Flatten the conv output
        x_flat = x.view(x.size(0), -1)  # Shape: [batch_size, 128*8*8]
        
        # Process through fully connected layer
        x = self.relu(self.fc1(x_flat))
        
        # Output policy logits
        policy_logits = self.policy_head(x)
        
        return policy_logits

def train_chess_network_from_preprocessed(
    data_dir,                  # Folder containing .npz files
    num_epochs=1,              # Number of complete passes through all data files
    batch_size=100,
    lr=0.0005,
    layers=5,
    checkpoint_interval=10000, # Save model every this many iterations
    pretrained_model_path=None,
    verbose=True
):
    """
    Train a simplified policy network using preprocessed samples stored in .npz files.
    """
    # Gather all .npz files in the folder.
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    if verbose:
        print(f"Found {len(npz_files)} .npz files in '{data_dir}'.")
    num_files = len(npz_files)
    if num_files == 0:
        raise ValueError(f"No .npz files found in {data_dir}")
    
    # Initialize a simplified model that doesn't use the evaluator
    model = SimplePolicyNet(layers=layers).to(device)
    
    if pretrained_model_path:
        # Load weights only for the matching layers
        pretrained_dict = torch.load(pretrained_model_path, map_location=device)
        model_dict = model.state_dict()
        
        # Filter out unnecessary keys
        filtered_dict = {k: v for k, v in pretrained_dict.items() 
                        if k in model_dict and model_dict[k].shape == v.shape}
        
        # Update model weights
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        
        if verbose:
            print(f"Loaded compatible pretrained weights from {pretrained_model_path}")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    metrics = {
        "losses": [],
        "iterations": 0,
        "samples_processed": 0,
        "epochs_completed": 0,
        "files_processed": 0
    }
    iteration = 0
    
    # Train for the specified number of epochs
    for epoch in range(num_epochs):
        if verbose:
            print(f"Starting epoch {epoch+1}/{num_epochs}")
        
        # Shuffle the file list at the beginning of each epoch
        random.shuffle(npz_files)
        
        # Process each file in the shuffled order
        for file_index, current_file in enumerate(npz_files):
            if verbose:
                print(f"Processing file {file_index+1}/{num_files}: {current_file}")
            try:
                # Load the file with batch processing
                checkpoint = np.load(current_file)
                
                # Extract data without immediately converting to PyTorch tensors
                states_np = checkpoint['states']
                moves_np = checkpoint['moves']
                
                # Get total number of samples in this file
                num_samples = len(states_np)
                indices = list(range(num_samples))
                random.shuffle(indices)
                sample_index = 0
                
                # Process mini-batches from the current file
                while sample_index + batch_size <= num_samples:
                    # Get batch indices
                    batch_indices = indices[sample_index: sample_index + batch_size]
                    
                    # Convert only the current batch to PyTorch tensors with proper types
                    batch_states = torch.from_numpy(states_np[batch_indices]).to(device).float()
                    batch_moves = torch.from_numpy(moves_np[batch_indices]).to(device).long()
                    
                    sample_index += batch_size
                    
                    # Update the model
                    optimizer.zero_grad()
                    policy_logits = model(batch_states)
                    loss = F.cross_entropy(policy_logits, batch_moves)
                    loss.backward()
                    optimizer.step()
                    
                    loss_value = loss.item()
                    
                    iteration += 1
                    metrics["losses"].append(loss_value)
                    metrics["iterations"] = iteration
                    metrics["samples_processed"] += batch_size
                    
                    if verbose and iteration % 100 == 0:
                        print(f"Iteration {iteration}, Loss: {loss_value:.4f}, Samples processed: {metrics['samples_processed']}")
                    
                    if iteration % checkpoint_interval == 0:
                        if verbose:
                            print(f"Checkpoint: {iteration} iterations processed. Saving model...")
                        save_models_and_metrics(policy_net=model, metrics=metrics, folder_name="simplified_real_data")
                
                # Process any leftover samples in the current file
                if sample_index < num_samples:
                    # Get batch indices
                    batch_indices = indices[sample_index:]
                    
                    # Convert only the remaining batch to PyTorch tensors with proper types
                    batch_states = torch.from_numpy(states_np[batch_indices]).to(device).float()
                    batch_moves = torch.from_numpy(moves_np[batch_indices]).to(device).long()
                    
                    # Update the model
                    optimizer.zero_grad()
                    policy_logits = model(batch_states)
                    loss = F.cross_entropy(policy_logits, batch_moves)
                    loss.backward()
                    optimizer.step()
                    
                    loss_value = loss.item()
                    
                    iteration += 1
                    metrics["losses"].append(loss_value)
                    metrics["iterations"] = iteration
                    metrics["samples_processed"] += len(batch_indices)
                    
                    if verbose and iteration % 100 == 0:
                        print(f"Iteration {iteration}, Loss: {loss_value:.4f}, Samples processed: {metrics['samples_processed']}")
                    
                    if iteration % checkpoint_interval == 0:
                        if verbose:
                            print(f"Checkpoint: {iteration} iterations processed. Saving model...")
                        save_models_and_metrics(policy_net=model, metrics=metrics, folder_name="simplified_real_data")
                
                metrics["files_processed"] += 1
                
            except Exception as e:
                print(f"Error loading {current_file}: {e}. Skipping this file.")
                continue
        
        # Update epoch tracking
        metrics["epochs_completed"] += 1
        if verbose:
            print(f"Completed epoch {epoch+1}/{num_epochs}")
            print(f"Total iterations: {iteration}, Total samples processed: {metrics['samples_processed']}")
    
    if verbose:
        print("Training complete. Saving final model...")
    save_models_and_metrics(policy_net=model, metrics=metrics, folder_name="simplified_real_data")
    
    return metrics, model

# Example usage
data_dir = r"pgn_checkpoints_compressed"
pretrained_model = None  # Or a valid path if you want to resume training.
metrics, trained_model = train_chess_network_from_preprocessed(
    data_dir=data_dir,
    num_epochs=5,              # Number of complete passes through all data files
    batch_size=124,
    lr=0.0001,
    layers=12,
    checkpoint_interval=200000,
    pretrained_model_path=pretrained_model,
    verbose=True
)