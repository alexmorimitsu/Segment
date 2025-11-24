#!/usr/bin/env python3
"""
Script to read all .npy files in Anotacoes directory and count non-zero unique values per file.
"""

import numpy as np
import os
import glob

def count_nonzero_unique_values(file_path):
    """Load a .npy file and count the number of non-zero unique values."""
    try:
        data = np.load(file_path)
        # Get unique values excluding zero
        unique_values = np.unique(data)
        nonzero_unique = unique_values[unique_values != 0]
        return len(nonzero_unique)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def main():
    # Directory containing .npy files
    anotacoes_dir = "/home/alexandre/IO2025/Data/Coral_Sol/Anotacoes"
    
    # Find all .npy files
    npy_files = glob.glob(os.path.join(anotacoes_dir, "*.npy"))
    npy_files.sort()  # Sort for consistent output
    
    print(f"Found {len(npy_files)} .npy files in {anotacoes_dir}")
    print("=" * 60)
    print(f"{'File Name':<40} {'Non-zero Unique Values':<20}")
    print("=" * 60)
    
    total_files = 0
    for file_path in npy_files:
        filename = os.path.basename(file_path)
        count = count_nonzero_unique_values(file_path)
        
        if count is not None:
            print(f"{filename:<40} {count:<20}")
            total_files += 1
        else:
            print(f"{filename:<40} {'ERROR':<20}")
    
    print("=" * 60)
    print(f"Successfully processed {total_files} files")

if __name__ == "__main__":
    main()
