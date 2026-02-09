import pickle
import torch
import sys

def check_for_nans(pkl_path):
    print(f"Checking for NaNs in {pkl_path}...")
    try:
        with open(pkl_path, 'rb') as f:
            all_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return

    total_valid = 0
    total_nan = 0
    
    for i, batch in enumerate(all_data):
        x_pred = batch.x_pred
        if torch.isnan(x_pred).any():
            print(f"❌ Batch {i} contains NaNs!")
            nans_in_batch = torch.isnan(x_pred).sum().item()
            total_size = x_pred.numel()
            print(f"   {nans_in_batch} / {total_size} values are NaN")
            total_nan += 1
        else:
            total_valid += 1
            
    print("-" * 30)
    if total_nan == 0:
        print("✅ SUCCESS: No NaNs found in any trajectory.")
        print(f"Checked {len(all_data)} batches.")
        
        # Print stats of first batch to show it looks normal
        first_batch = all_data[0]
        print("\nSample Data Stats (Batch 0):")
        print(f"Min value: {first_batch.x_pred.min():.4f}")
        print(f"Max value: {first_batch.x_pred.max():.4f}")
        print(f"Mean value: {first_batch.x_pred.mean():.4f}")
    else:
        print(f"⚠️ FAILURE: Found NaNs in {total_nan} batches.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_for_nans(sys.argv[1])
    else:
        print("Please provide path to samples.pkl")
