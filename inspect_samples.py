#!/usr/bin/env python3
"""
Load and inspect sampled MD17 trajectories from GeoTDM.
"""
import pickle
import torch
import numpy as np
import sys

def load_and_inspect_samples(samples_path):
    """Load and inspect the sampled trajectories."""
    print(f"Loading samples from: {samples_path}")
    
    with open(samples_path, 'rb') as f:
        all_data = pickle.load(f)
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Number of batches: {len(all_data)}")
    
    # Inspect first batch
    first_batch = all_data[0]
    print(f"\nFirst batch structure:")
    print(f"  Keys: {first_batch.keys}")
    
    x_true = first_batch.x  # Ground truth
    x_pred = first_batch.x_pred  # Sampled trajectory
    
    print(f"\n{'='*80}")
    print(f"TRAJECTORY SHAPES")
    print(f"{'='*80}")
    print(f"Ground truth (x):     {x_true.shape}")
    print(f"Sampled (x_pred):     {x_pred.shape}")
    print(f"  Format: [N_atoms, 3_coords, T_timesteps]")
    
    # Calculate statistics
    print(f"\n{'='*80}")
    print(f"STATISTICS (First Batch)")
    print(f"{'='*80}")
    
    # RMSD between ground truth and prediction
    diff = x_true - x_pred
    rmsd_per_frame = torch.sqrt((diff ** 2).sum(dim=(0, 1)) / x_true.shape[0])
    
    print(f"\nRMSD per timestep (Angstroms):")
    print(f"  Mean:   {rmsd_per_frame.mean():.4f}")
    print(f"  Std:    {rmsd_per_frame.std():.4f}")
    print(f"  Min:    {rmsd_per_frame.min():.4f}")
    print(f"  Max:    {rmsd_per_frame.max():.4f}")
    
    # Count total samples across all batches
    total_samples = 0
    for batch in all_data:
        batch_size = batch.batch.max().item() + 1
        total_samples += batch_size
    
    print(f"\n{'='*80}")
    print(f"Total number of sampled trajectories: {total_samples}")
    print(f"{'='*80}")
    
    return all_data

def save_trajectory_to_xyz(data, batch_idx, output_path, atom_types=None):
    """
    Save a single trajectory to XYZ format for visualization.
    
    Args:
        data: PyTorch Geometric Data object
        batch_idx: Which trajectory in the batch to save
        output_path: Path to save XYZ file
        atom_types: List of atom type symbols (e.g., ['C', 'H', 'O'])
    """
    x_pred = data.x_pred  # [N, 3, T]
    batch = data.batch
    
    # Extract trajectory for specific batch index
    mask = (batch == batch_idx)
    traj = x_pred[mask]  # [N_atoms, 3, T]
    
    n_atoms = traj.shape[0]
    n_frames = traj.shape[2]
    
    # Default atom types if not provided
    if atom_types is None:
        atom_types = ['C'] * n_atoms
    
    with open(output_path, 'w') as f:
        for t in range(n_frames):
            f.write(f"{n_atoms}\n")
            f.write(f"Frame {t}\n")
            for i in range(n_atoms):
                x, y, z = traj[i, :, t].numpy()
                f.write(f"{atom_types[i]} {x:.6f} {y:.6f} {z:.6f}\n")
    
    print(f"Saved trajectory to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_samples.py <samples.pkl> [--save-xyz <output.xyz>]")
        sys.exit(1)
    
    samples_path = sys.argv[1]
    all_data = load_and_inspect_samples(samples_path)
    
    # Optional: save first trajectory to XYZ
    if len(sys.argv) >= 4 and sys.argv[2] == '--save-xyz':
        output_xyz = sys.argv[3]
        print(f"\nSaving first trajectory to XYZ format...")
        save_trajectory_to_xyz(all_data[0], batch_idx=0, output_path=output_xyz)
        print(f"\nYou can visualize this with tools like VMD, PyMOL, or Ovito:")
        print(f"  vmd {output_xyz}")
