import os
import chess
from ChessRL import ChessPolicyNet
import random
import os
from itertools import cycle
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import glob
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use the first available GPU
    print("GPU is available and being used.")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU instead.")
import numpy as np

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
    
    Parameters:
        data (list or np.array): The data to be binned.
        num_bins (int): Number of bins to aggregate the data into.
    
    Returns:
        bin_centers (np.array): The x-axis positions (bin centers).
        bin_means (np.array): The mean value in each bin.
        bin_stds (np.array): The standard deviation in each bin.
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
    
    This version is adapted for supervised training. It expects the metrics dictionary
    to contain a key 'losses' (a list of loss values from training iterations) and plots
    the binned average loss vs. iteration number.
    
    Parameters:
        metrics (dict): Dictionary containing training metrics. Must include 'losses'.
        num_bins (int): Number of bins into which the loss data will be aggregated.
    
    Returns:
        fig (matplotlib.figure.Figure): The figure object containing the plot.
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

def update_on_batch(agent, optimizer, batch_samples):
    """
    Performs a supervised update on a batch of samples.
    Each sample is a tuple (state_tensor, move_index).
    """
    # Stack state tensors into a single tensor of shape (B, 12, 8, 8)
    states = torch.stack([sample[0] for sample in batch_samples]).to(device)
    targets = torch.tensor([sample[1] for sample in batch_samples], dtype=torch.long, device=device)
    optimizer.zero_grad()
    x = agent.conv_layers(states)
    x = x.view(x.size(0), -1)
    x = F.relu(agent.fc1(x))
    logits = agent.fc2(x)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_chess_network_from_preprocessed(
    data_dir,                  # Folder containing .pt files
    total_iterations=120000,   # Total mini-batch updates
    batch_size=100,
    lr=0.0005,
    layers=5,
    checkpoint_interval=10000, # Save model every this many iterations
    pretrained_model_path=None,
    verbose=True
):
    """
    Train a ChessPolicyNet model using preprocessed samples stored in .pt files.
    
    Each .pt file in data_dir is expected to contain a list of tuples:
        (state_tensor, UCI_move, move_index, result)
    
    For training, only the (state_tensor, move_index) pair is used.
    
    Instead of loading all files at once, this function loads one file at a time,
    processes its samples in mini-batches, then moves on to the next file.
    The files are processed in random order and reshuffled once all files are done.
    
    Parameters:
      data_dir (str): Folder containing the preprocessed .pt files.
      total_iterations (int): Total training iterations (mini-batch updates).
      batch_size (int): Mini-batch size.
      lr (float): Learning rate.
      layers (int): Number of convolutional layers for the model.
      checkpoint_interval (int): Save checkpoint every this many iterations.
      pretrained_model_path (str or None): Path to a pretrained model (if any).
      verbose (bool): If True, prints progress messages.
      
    Returns:
      metrics (dict): Training metrics.
    """
    # Gather all .pt files in the folder.
    pt_files = glob.glob(os.path.join(data_dir, "*.pt"))
    if verbose:
        print(f"Found {len(pt_files)} .pt files in '{data_dir}'.")
    # Shuffle the file list randomly.
    random.shuffle(pt_files)
    num_files = len(pt_files)
    
    # Initialize the model.
    model = ChessPolicyNet(
        board=chess.Board(),
        color=chess.WHITE,  # Arbitrary in a single-model setup
        device=device,
        layers=layers,
        epsilon=0.0      # No exploration during supervised training.
    ).to(device)
    
    if pretrained_model_path:
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device,weights_only=True))
        if verbose:
            print(f"Loaded pretrained weights from {pretrained_model_path}")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    metrics = {
        "losses": [],
        "iterations": 0,
        "samples_processed": 0
    }
    iteration = 0
    
    file_index = 0  # Pointer to current file in the list.
    
    while iteration < total_iterations:
        # If we've processed all files, reshuffle and restart.
        if file_index >= num_files:
            random.shuffle(pt_files)
            file_index = 0
        
        current_file = pt_files[file_index]
        file_index += 1
        
        if verbose:
            print(f"Processing file: {current_file}")
        try:
            data = torch.load(current_file,weights_only=True)
        except Exception as e:
            print(f"Error loading {current_file}: {e}. Skipping this file.")
            continue
        
        # Each file should contain a list of tuples:
        # (state_tensor, UCI_move, move_index, result)
        # For training, extract only (state_tensor, move_index)
        training_samples = [(state, move_idx) for (state, uci_move, move_idx, result) in data]
        random.shuffle(training_samples)
        num_samples = len(training_samples)
        sample_index = 0
        
        # Process mini-batches from the current file.
        while sample_index + batch_size <= num_samples and iteration < total_iterations:
            batch_samples = training_samples[sample_index: sample_index + batch_size]
            sample_index += batch_size
            loss = update_on_batch(model, optimizer, batch_samples)
            iteration += 1
            metrics["losses"].append(loss)
            metrics["iterations"] = iteration
            metrics["samples_processed"] += batch_size
            
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}/{total_iterations}, Loss: {loss:.4f}, Samples processed: {metrics['samples_processed']}")
            
            if iteration % checkpoint_interval == 0:
                if verbose:
                    print(f"Checkpoint: {iteration} iterations processed. Saving model...")
                save_models_and_metrics(policy_net=model, metrics=metrics, folder_name="real_data")
        
        # Process any leftover samples in the current file.
        if sample_index < num_samples and iteration < total_iterations:
            batch_samples = training_samples[sample_index:]
            loss = update_on_batch(model, optimizer, batch_samples)
            iteration += 1
            metrics["losses"].append(loss)
            metrics["iterations"] = iteration
            metrics["samples_processed"] += len(batch_samples)
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}/{total_iterations}, Loss: {loss:.4f}, Samples processed: {metrics['samples_processed']}")
            if iteration % checkpoint_interval == 0:
                if verbose:
                    print(f"Checkpoint: {iteration} iterations processed. Saving model...")
                save_models_and_metrics(policy_net=model, metrics=metrics, folder_name="real_data")
    
    if verbose:
        print("Training complete. Saving final model...")
    save_models_and_metrics(policy_net=model, metrics=metrics, folder_name="real_data")
    
    return metrics

# Replace with the path to your preprocessed .pt file.
data_dir = r"pgn_checkpoints"
pretrained_model = "real_data11\policy_net.pth"  # Or a valid path if you want to resume training.
metrics = train_chess_network_from_preprocessed(
    data_dir=data_dir,
    total_iterations=100000000,
    batch_size=64,
    lr=0.005,
    layers=2,
    checkpoint_interval=100000,
    pretrained_model_path=pretrained_model,
    verbose=True
)