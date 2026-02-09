#!/usr/bin/env python3
"""
Convert sampled MD17 trajectories to XYZ format.
Exports both ground truth and sampled trajectories.
Uses actual atomic numbers from the dataset file.
"""
import pickle
import torch
import numpy as np
import os
import sys
from tqdm import tqdm

def get_atom_types_from_npz(npz_path):
    """
    Load atomic numbers from NPZ file and convert to symbols.
    """
    print(f"Loading atomic numbers from: {npz_path}")
    try:
        data = np.load(npz_path)
        z = data['z']
        
        # Atomic number to symbol map
        periodic_table = {
            1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 
            15: 'P', 16: 'S', 17: 'Cl'
        }
        
        atom_types = [periodic_table.get(atomic_num, 'X') for atomic_num in z]
        print(f"Found {len(atom_types)} atoms: {atom_types}")
        return atom_types
    except Exception as e:
        print(f"Error loading NPZ: {e}")
        sys.exit(1)

def save_trajectory_to_xyz(coords, output_path, atom_types, comment=""):
    """
    Save a single trajectory to XYZ format.
    
    Args:
        coords: Trajectory coordinates [N_atoms, 3, T_timesteps]
        output_path: Path to save XYZ file
        atom_types: List of atom symbols
        comment: Comment line for each frame
    """
    n_atoms = coords.shape[0]
    n_frames = coords.shape[2]
    
    if len(atom_types) != n_atoms:
        print(f"Error: Number of atoms ({n_atoms}) matches atom types ({len(atom_types)})")
        return
    
    with open(output_path, 'w') as f:
        for t in range(n_frames):
            f.write(f"{n_atoms}\n")
            f.write(f"{comment} Frame {t}\n")
            for i in range(n_atoms):
                x, y, z = coords[i, :, t]
                f.write(f"{atom_types[i]:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")

def convert_samples_to_xyz(samples_path, output_dir, npz_path, max_trajectories=None):
    """
    Convert all sampled trajectories to XYZ files.
    """
    # 1. Get correct atom types first
    atom_types = get_atom_types_from_npz(npz_path)
    
    # 2. Load samples
    print(f"Loading samples from: {samples_path}")
    with open(samples_path, 'rb') as f:
        all_data = pickle.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    gt_dir = os.path.join(output_dir, 'ground_truth')
    pred_dir = os.path.join(output_dir, 'sampled')
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    
    traj_count = 0
    
    print(f"\nConverting trajectories to XYZ format...")
    print(f"Output directory: {output_dir}")
    print(f"Using atom types: {' '.join(atom_types)}")
    
    for batch_idx, batch in enumerate(all_data):
        x_true = batch.x.cpu()       # [N_total, 3, T]
        x_pred = batch.x_pred.cpu()  # [N_total, 3, T]
        batch_indices = batch.batch.cpu()
        
        # Get number of trajectories in this batch
        n_traj_in_batch = batch_indices.max().item() + 1
        
        for traj_idx in range(n_traj_in_batch):
            if max_trajectories is not None and traj_count >= max_trajectories:
                print(f"\nReached maximum of {max_trajectories} trajectories.")
                return traj_count
            
            # Extract atoms for this trajectory
            mask = (batch_indices == traj_idx)
            traj_gt = x_true[mask]    # [N_atoms, 3, T]
            traj_pred = x_pred[mask]  # [N_atoms, 3, T]
            
            # Check size match
            if traj_gt.shape[0] != len(atom_types):
                print(f"Warning: Molecule size mismatch! Data has {traj_gt.shape[0]}, expected {len(atom_types)}")
                continue

            # Save ground truth
            gt_path = os.path.join(gt_dir, f'trajectory_{traj_count:04d}_gt.xyz')
            save_trajectory_to_xyz(traj_gt.numpy(), gt_path, 
                                  atom_types=atom_types,
                                  comment=f"Aspirin MD17 Ground Truth {traj_count}")
            
            # Save sampled
            pred_path = os.path.join(pred_dir, f'trajectory_{traj_count:04d}_sampled.xyz')
            save_trajectory_to_xyz(traj_pred.numpy(), pred_path,
                                  atom_types=atom_types,
                                  comment=f"Aspirin MD17 Sampled {traj_count}")
            
            traj_count += 1
            
            if traj_count % 100 == 0:
                print(f"Converted {traj_count} trajectories...")
    
    print(f"\n✅ Conversion complete!")
    print(f"Total trajectories converted: {traj_count}")
    print(f"Ground truth files: {gt_dir}/")
    print(f"Sampled files: {pred_dir}/")
    
    return traj_count

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert sampled trajectories to XYZ format')
    parser.add_argument('samples_pkl', type=str, help='Path to samples.pkl file')
    parser.add_argument('--output-dir', type=str, default='xyz_valid',
                       help='Output directory for XYZ files')
    parser.add_argument('--npz-path', type=str, default='data/md17/md17_aspirin.npz',
                       help='Path to original NPZ file for atom types')
    parser.add_argument('--max-traj', type=int, default=None,
                       help='Maximum number of trajectories to convert (default: all)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.npz_path):
        print(f"Error: NPZ file not found at {args.npz_path}")
        print("Please check the path to your MD17 data.")
        sys.exit(1)
        
    convert_samples_to_xyz(args.samples_pkl, args.output_dir, args.npz_path, args.max_traj)
