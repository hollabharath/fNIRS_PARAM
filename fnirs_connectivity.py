#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fNIRS Resting-State Connectivity Pipeline

Description:
This script runs a full preprocessing pipeline on fNIRS data for resting-state
connectivity analysis. It takes a directory of .snirf files, performs quality
control (SCI), automatically applies short-channel regression if available,
corrects for motion artifacts (TDDR), filters the data, and calculates a
channel-wise correlation matrix for the HbO signal. Bad channels are marked
as NA in the final output.

Author:
Dr. Bharath Holla, hollabharath@gmail.com

Date:
July 17, 2025
"""

import os
import sys
import glob
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import compress

def check_requirements():
    """Checks if all required packages are installed."""
    required_packages = ['mne', 'mne_nirs', 'pandas', 'matplotlib', 'seaborn']
    missing_packages = []
    print("Checking for required packages...")
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("\nError: The following required packages are not installed:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nPlease install the missing packages to proceed.")
        print(f"Example command: pip install {' '.join(missing_packages)}")
        sys.exit(1)
    else:
        print("All requirements are satisfied.")

# Now import the necessary libraries
import mne
import seaborn as sns
import mne_nirs
from mne.preprocessing.nirs import (
    optical_density,
    scalp_coupling_index,
    source_detector_distances,
    temporal_derivative_distribution_repair,
    beer_lambert_law
)
from mne_nirs.signal_enhancement import short_channel_regression

def get_canonical_name(ch_name):
    """Extracts the base source-detector name (e.g., 'S1_D1') from a full channel name."""
    return ch_name.split(" ")[0]

def preprocess_and_generate_connectivity(snirf_file, output_dir, sci_threshold=0.5):
    """
    Runs a preprocessing and connectivity pipeline on a single fNIRS file.
    """
    print(f"Processing: {snirf_file}")
    basename = os.path.splitext(os.path.basename(snirf_file))[0]

    # --- 1. Load Data & Initial QC ---
    try:
        raw_intensity = mne.io.read_raw_snirf(snirf_file, preload=True)
    except Exception as e:
        print(f"  Error loading file: {e}")
        return

    raw_od = optical_density(raw_intensity)
    sci = scalp_coupling_index(raw_od)
    
    bad_od_channels = list(compress(raw_od.ch_names, sci < sci_threshold))
    bad_canonical_names = {get_canonical_name(name) for name in bad_od_channels}
    print(f"  Identified {len(bad_canonical_names)} bad channel pairs based on SCI < {sci_threshold}")

    # --- 2. Preprocessing ---
    
    # Check for short channels
    distances = source_detector_distances(raw_od.info)
    
    # If short channels exist, apply regression
    if np.any(distances < 0.01):
        print("  Short channels found. Applying short-channel regression...")
        raw_od_corrected_1 = short_channel_regression(raw_od)
    else:
        print("  No short channels found. Skipping short-channel regression.")
        raw_od_corrected_1 = raw_od # Pass data through unchanged
        
    # Apply TDDR for further motion correction
    print("  Applying TDDR for motion correction...")
    raw_od_corrected_2 = temporal_derivative_distribution_repair(raw_od_corrected_1)
    
    # Convert to hemoglobin and filter
    raw_haemo = beer_lambert_law(raw_od_corrected_2, ppf=6.0)
    print("  Applying band-pass filter (0.01 - 0.8 Hz)...")
    raw_haemo.filter(0.01, 0.8, l_trans_bandwidth='auto', h_trans_bandwidth='auto')

    # --- 3. Connectivity Calculation ---
    hbo_channel_indices = mne.pick_types(raw_haemo.info, fnirs='hbo')
    all_hbo_names = [raw_haemo.ch_names[i] for i in hbo_channel_indices]
    all_hbo_canonical_names = [get_canonical_name(name) for name in all_hbo_names]

    if len(hbo_channel_indices) < 2:
        print("  Not enough channels to calculate connectivity. Skipping.")
        return

    print(f"  Calculating full connectivity matrix...")
    hbo_data = raw_haemo.get_data(picks=hbo_channel_indices)
    correlation_matrix = np.corrcoef(hbo_data)
    
    bad_indices = [i for i, name in enumerate(all_hbo_canonical_names) if name in bad_canonical_names]
    for bad_idx in bad_indices:
        correlation_matrix[bad_idx, :] = np.nan
        correlation_matrix[:, bad_idx] = np.nan
    
    # --- 4. Save Outputs ---
    df_corr = pd.DataFrame(correlation_matrix, index=all_hbo_canonical_names, columns=all_hbo_canonical_names)
    csv_path = os.path.join(output_dir, f"{basename}_connectivity_matrix.csv")
    df_corr.to_csv(csv_path, na_rep='NA')
    print(f"  -> Connectivity matrix saved to: {os.path.basename(csv_path)}")

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df_corr, cmap='coolwarm',vmin=-1, vmax=1, ax=ax, annot=False, cbar_kws={'label': 'Pearson Correlation'})
    ax.set_title(f"Connectivity Heatmap (HbO) - {basename}")
    fig_path = os.path.join(output_dir, f"{basename}_connectivity_heatmap.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Connectivity heatmap saved to: {os.path.basename(fig_path)}")


def run_connectivity_pipeline(input_directory):
    """Finds and processes all .snirf files in a directory."""
    snirf_files = glob.glob(os.path.join(input_directory, '**', '*.snirf'), recursive=True)
    if not snirf_files:
        print(f"No .snirf files found in '{input_directory}'")
        return

    parent_dir = os.path.dirname(os.path.abspath(input_directory))
    output_directory = os.path.join(parent_dir, "fnirs_connectivity_results")
    os.makedirs(output_directory, exist_ok=True)
    print(f"All connectivity results will be saved in: {output_directory}")

    for snirf_file in snirf_files:
        preprocess_and_generate_connectivity(snirf_file, output_directory)
        print("-" * 50)
    
    print("\n--- Pipeline Complete ---")


if __name__ == "__main__":
    # Run a check for required packages first
    check_requirements()
    
    # Set Matplotlib backend to avoid display errors when run from terminal
    plt.switch_backend('Agg')

    if len(sys.argv) != 2:
        print("\nUsage: python fnirs_connectivity.py <input_directory>")
        sys.exit(1)

    input_dir_path = sys.argv[1]

    if not os.path.isdir(input_dir_path):
        print(f"\nError: Input directory not found: {input_dir_path}")
        sys.exit(1)
        
    run_connectivity_pipeline(input_dir_path)