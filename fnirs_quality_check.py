#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fNIRS Signal Quality Pipeline

Description:
This script automates the signal quality assessment for fNIRS data stored in the
.snirf format. It processes all .snirf files found within a specified input
directory, generating a multi-page PDF report for each file and two summary
CSV files in a wide format for easy analysis.

Author:
Dr. Bharath Holla, hollabharath@gmail.com

Date:
July 16, 2025
"""

import os
import glob
import sys
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import compress

# This check runs before attempting to import MNE, which is slow.
def check_requirements():
    """Checks if all required packages are installed."""
    required_packages = ['mne', 'mne_nirs', 'pandas', 'matplotlib']
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

# Now import the heavy libraries
import mne
from mne.preprocessing.nirs import optical_density, scalp_coupling_index, beer_lambert_law
import mne_nirs
from mne_nirs.visualisation import plot_timechannel_quality_metric
from mne_nirs.preprocessing import scalp_coupling_index_windowed
from matplotlib.backends.backend_pdf import PdfPages

def process_snirf_file(snirf_file, output_pdf_dir, sci_threshold=0.5):
    """
    Processes a single SNIRF file to generate a PDF report and return channel-wise SCI metrics.
    """
    print(f"Processing: {snirf_file}")
    basename = os.path.splitext(os.path.basename(snirf_file))[0]

    try:
        raw_intensity = mne.io.read_raw_snirf(snirf_file, preload=True)
        raw_od = optical_density(raw_intensity)
    except Exception as e:
        print(f"  Error loading {snirf_file}: {e}")
        return None

    ch_names = raw_od.ch_names

    # --- SCI Quality Assessment ---
    overall_sci = scalp_coupling_index(raw_od)
    try:
        _, scores_sci, times_sci = scalp_coupling_index_windowed(raw_od, time_window=60)
        percent_good_sci_windows = np.sum(scores_sci > sci_threshold, axis=1) / scores_sci.shape[1] * 100
    except Exception as e:
        print(f"  Warning: Could not compute windowed SCI for {basename}. Skipping related plots/metrics. Error: {e}")
        scores_sci = np.full((len(ch_names), 1), np.nan)
        percent_good_sci_windows = np.full(len(ch_names), np.nan)
        times_sci = np.array([0])

    # --- Prepare Channel-wise DataFrame for this file ---
    df_channel = pd.DataFrame({
        'subject_id': basename,
        'channel': ch_names,
        'overall_sci': np.round(overall_sci, 4),
        'percent_good_sci_windows': np.round(percent_good_sci_windows, 2),
    })

    # --- PDF Report Generation ---
    output_pdf_filename = os.path.join(output_pdf_dir, f"fnirs_quality_report_{basename}.pdf")
    bad_channels = list(compress(ch_names, overall_sci < sci_threshold))
    raw_od.info["bads"] = bad_channels

    with PdfPages(output_pdf_filename) as pdf:
        # Page 1: SCI Histogram
        fig1, ax = plt.subplots()
        ax.hist(overall_sci, bins=20)
        ax.set(xlabel="Scalp Coupling Index", ylabel="Count", xlim=[0, 1])
        ax.axvline(sci_threshold, color='r', linestyle='--', label=f'Threshold ({sci_threshold})')
        ax.legend()
        pdf.savefig(fig1)
        plt.close(fig1)

        # Page 2: Raw Optical Density
        fig2 = raw_od.plot(duration=300, show_scrollbars=False, clipping=None, n_channels=40)
        pdf.savefig(fig2)
        plt.close(fig2)

        # Apply Beer-Lambert law and filter
        raw_haemo = beer_lambert_law(raw_od, ppf=0.1)
        raw_haemo.filter(0.01, 0.8, l_trans_bandwidth='auto', h_trans_bandwidth='auto')

        # Page 3: Filtered Hemoglobin Data
        fig3 = raw_haemo.plot(n_channels=len(raw_haemo.ch_names), duration=300, show_scrollbars=False)
        pdf.savefig(fig3)
        plt.close(fig3)

        # Page 4: Sliding Window SCI Plot
        if not np.isnan(scores_sci).all():
             fig4 = plot_timechannel_quality_metric(raw_od, scores_sci, times_sci, threshold=sci_threshold, title="Scalp Coupling Index Quality Evaluation")
             pdf.savefig(fig4)
             plt.close(fig4)

        # Page 5: Bad Channels on Scalp
        fig5 = raw_od.plot_sensors(ch_type='fnirs_od')
        pdf.savefig(fig5)
        plt.close(fig5)

    return df_channel


def run_pipeline(input_directory):
    """Finds and processes all .snirf files, generating reports."""
    snirf_files = glob.glob(os.path.join(input_directory, '**', '*.snirf'), recursive=True)
    if not snirf_files:
        print(f"No .snirf files found in {input_directory}")
        return

    parent_dir = os.path.dirname(os.path.abspath(input_directory))
    main_output_dir = os.path.join(parent_dir, 'fnirs_quality_reports')
    os.makedirs(main_output_dir, exist_ok=True)
    
    pdf_output_dir = os.path.join(main_output_dir, 'fnirs_pdf_reports')
    os.makedirs(pdf_output_dir, exist_ok=True)
    print(f"All reports will be saved in: {main_output_dir}")

    all_channel_dfs = [df for snirf_file in snirf_files if (df := process_snirf_file(snirf_file, pdf_output_dir)) is not None]

    if all_channel_dfs:
        final_long_df = pd.concat(all_channel_dfs, ignore_index=True)

        avg_sci_wide = final_long_df.pivot_table(
            index='subject_id', columns='channel', values='overall_sci'
        )
        avg_sci_path = os.path.join(main_output_dir, "sci_average_report.csv")
        avg_sci_wide.to_csv(avg_sci_path)
        print(f"\nWide-format average SCI report saved to: {avg_sci_path}")

        percent_good_wide = final_long_df.pivot_table(
            index='subject_id', columns='channel', values='percent_good_sci_windows'
        )
        if not percent_good_wide.empty and not percent_good_wide.isnull().all().all():
            percent_good_path = os.path.join(main_output_dir, "sci_percent_good_report.csv")
            percent_good_wide.to_csv(percent_good_path)
            print(f"Wide-format percent good SCI report saved to: {percent_good_path}")
        else:
            print("Skipping 'sci_percent_good_report.csv' because no valid windowed SCI data was computed.")

        print("\n--- Pipeline Complete ---")
    else:
        print("\n--- Pipeline Complete ---\nNo data was processed successfully.")


if __name__ == "__main__":
    # Run a check for required packages first
    check_requirements()
    
    # Set Matplotlib backend after imports and checks
    plt.switch_backend('Agg')

    if len(sys.argv) != 2:
        print("\nUsage: python your_script_name.py <input_directory>")
        sys.exit(1)
        
    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        print(f"\nError: Directory not found: {input_dir}")
        sys.exit(1)
        
    run_pipeline(input_dir)