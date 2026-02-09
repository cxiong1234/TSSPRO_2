#!/usr/bin/env python3
"""
Compare two PyTorch checkpoint files to check if they are identical or similar.
"""
import torch
import sys
import numpy as np

def compare_checkpoints(ckpt1_path, ckpt2_path):
    """Compare two checkpoint files."""
    print(f"Loading checkpoint 1: {ckpt1_path}")
    ckpt1 = torch.load(ckpt1_path, map_location='cpu')
    
    print(f"Loading checkpoint 2: {ckpt2_path}")
    ckpt2 = torch.load(ckpt2_path, map_location='cpu')
    
    # Check if they are exactly the same
    if isinstance(ckpt1, dict) and isinstance(ckpt2, dict):
        keys1 = set(ckpt1.keys())
        keys2 = set(ckpt2.keys())
        
        print(f"\nCheckpoint 1 has {len(keys1)} keys")
        print(f"Checkpoint 2 has {len(keys2)} keys")
        
        if keys1 != keys2:
            print("\n❌ Checkpoints have different keys!")
            print(f"Keys only in ckpt1: {keys1 - keys2}")
            print(f"Keys only in ckpt2: {keys2 - keys1}")
            return
        
        # Compare each parameter
        identical = True
        max_diff = 0.0
        total_params = 0
        
        print("\nComparing parameters:")
        print("-" * 80)
        
        for key in keys1:
            param1 = ckpt1[key]
            param2 = ckpt2[key]
            
            if isinstance(param1, torch.Tensor) and isinstance(param2, torch.Tensor):
                if param1.shape != param2.shape:
                    print(f"❌ {key}: Different shapes {param1.shape} vs {param2.shape}")
                    identical = False
                else:
                    diff = torch.abs(param1 - param2)
                    max_param_diff = diff.max().item()
                    mean_param_diff = diff.mean().item()
                    
                    total_params += param1.numel()
                    max_diff = max(max_diff, max_param_diff)
                    
                    if max_param_diff > 0:
                        identical = False
                        print(f"  {key:50s} | max_diff: {max_param_diff:.6e} | mean_diff: {mean_param_diff:.6e}")
                    else:
                        print(f"✓ {key:50s} | IDENTICAL")
            else:
                if param1 != param2:
                    print(f"❌ {key}: Different values")
                    identical = False
                else:
                    print(f"✓ {key}: IDENTICAL")
        
        print("-" * 80)
        print(f"\nTotal parameters: {total_params:,}")
        
        if identical:
            print("\n✅ The checkpoints are IDENTICAL!")
        else:
            print(f"\n⚠️  The checkpoints are DIFFERENT")
            print(f"Maximum absolute difference: {max_diff:.6e}")
            
            # Calculate similarity percentage
            all_diffs = []
            for key in keys1:
                param1 = ckpt1[key]
                param2 = ckpt2[key]
                if isinstance(param1, torch.Tensor) and isinstance(param2, torch.Tensor):
                    if param1.shape == param2.shape:
                        all_diffs.append(torch.abs(param1 - param2).flatten())
            
            if all_diffs:
                all_diffs = torch.cat(all_diffs)
                mean_diff = all_diffs.mean().item()
                median_diff = all_diffs.median().item()
                
                print(f"Mean absolute difference: {mean_diff:.6e}")
                print(f"Median absolute difference: {median_diff:.6e}")
    else:
        print("Checkpoints are not dictionaries, comparing directly...")
        if torch.equal(ckpt1, ckpt2):
            print("✅ The checkpoints are IDENTICAL!")
        else:
            print("❌ The checkpoints are DIFFERENT")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_checkpoints.py <checkpoint1.pt> <checkpoint2.pt>")
        sys.exit(1)
    
    compare_checkpoints(sys.argv[1], sys.argv[2])
