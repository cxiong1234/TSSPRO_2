import numpy as np
import pandas as pd
import sys
import hashlib

def get_atomic_number(atom_name):
    """
    Deduce atomic number from PDB atom name.
    """
    # Clean up atom name (remove numbers, strip whitespace)
    atom_name = atom_name.strip()
    
    # Special cases handling based on common PDB conventions
    if atom_name.startswith('C'): return 6   # Carbon
    if atom_name.startswith('H'): return 1   # Hydrogen
    if atom_name.startswith('Q'): return 1   # Hydrogen placeholder often used in AMBER/CHARMM (QQH)
    if atom_name.startswith('O'): return 8   # Oxygen
    if atom_name.startswith('N'): return 7   # Nitrogen
    if atom_name.startswith('S'): return 16  # Sulfur
    if atom_name.startswith('P'): return 15  # Phosphorus
    if atom_name.startswith('F') and atom_name != 'FAD': return 9 # Fluorine (unlikely here but good for completeness)
    
    # 1-letter fallback (e.g. for simple element names)
    element = atom_name[0]
    table = {'C': 6, 'H': 1, 'O': 8, 'N': 7, 'S': 16, 'P': 15, 'F': 9}
    return table.get(element, 0) # 0 for unknown

def convert_qm_data(coords_path, atom_index_path, output_path):
    print(f"Reading coordinates from {coords_path}...")
    R = np.load(coords_path)
    print(f"Coordinates shape: {R.shape}")
    
    print(f"Reading atom info from {atom_index_path}...")
    # Parse the fixed-width or space-separated text file
    # Format: index resid resname atomname
    # Skip header line if present
    atoms = []
    with open(atom_index_path, 'r') as f:
        lines = f.readlines()
        
    z_list = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        parts = line.split()
        if len(parts) < 4:
            continue
            
        # parts[3] is atomname
        atom_name = parts[3]
        z = get_atomic_number(atom_name)
        z_list.append(z)
        atoms.append(atom_name)
        
    z_array = np.array(z_list, dtype=np.uint8)
    print(f"Extracted {len(z_array)} atoms.")
    print(f"Atom types found: {np.unique(z_array)}")
    
    if len(z_array) != R.shape[1]:
        print(f"ERROR: Mismatch between number of atoms in Coords ({R.shape[1]}) and Index file ({len(z_array)})!")
        sys.exit(1)

    # placeholder data
    n_frames = R.shape[0]
    n_atoms = R.shape[1]
    
    E = np.zeros((n_frames, 1)) # Energy
    F = np.zeros((n_frames, n_atoms, 3)) # Forces
    
    # Create dictionary for npz
    data = {
        'R': R,
        'z': z_array,
        'E': E,
        'F': F,
        'name': np.array(['QM_region']),
        'theory': np.array(['unknown']),
        'type': np.array(['unknown']),
        'md5': np.array(['unknown'])
    }
    
    print(f"Saving to {output_path}...")
    np.savez_compressed(output_path, **data)
    print("Done!")

if __name__ == "__main__":
    coords_file = 'data/md17/QM_region_coords.npy'
    atom_file = 'data/md17/QM_region_atom_index.txt'
    output_file = 'data/md17/md17_qm_region.npz'
    
    convert_qm_data(coords_file, atom_file, output_file)
