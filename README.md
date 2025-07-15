# fNIRS Preprocessing & Analysis Pipelines

This repository contains scripts for the preprocessing and analysis of resting-state functional near-infrared spectroscopy (fNIRS) data in the `.snirf` format.

-   **`fnirs_quality_check.py`**: A pipeline for assessing signal quality.
-   **`fnirs_connectivity.py`**: A pipeline for preprocessing and calculating resting-state functional connectivity.

---

## 1. Signal Quality Pipeline (`fnirs_quality_check.py`)

This script automates the signal quality assessment for fNIRS data. It iterates through all `.snirf` files in a specified directory and generates detailed reports.

### Features

-   Calculates the Scalp Coupling Index (SCI) as a primary quality metric.
-   Generates a multi-page PDF report for each subject, including:
    -   Histogram of overall SCI values.
    -   Plot of raw optical density data.
    -   Plot of filtered hemoglobin data (HbO/HbR).
    -   Visualization of SCI values over sliding windows.
    -   Scalp map showing the location of bad channels.
-   Outputs two summary CSV files in a wide format for easy group-level analysis:
    -   `sci_average_report.csv`: Contains the overall SCI for every channel for all subjects.
    -   `sci_percent_good_report.csv`: Contains the percentage of time windows with good SCI for every channel for all subjects.

### Usage

To run the quality pipeline, execute the script from your terminal and provide the path to the directory containing your `.snirf` files.

```bash
python fnirs_quality_check.py /path/to/your/data_directory
```

The script will create a new directory named `fnirs_quality_reports` in the parent directory of your data folder. This new directory will contain the PDF reports and summary CSVs.

---

## 2. Resting-State Connectivity Pipeline (`fnirs_connectivity.py`)

This script runs a complete preprocessing pipeline on resting-state fNIRS data and calculates functional connectivity.

### Features

-   Performs initial quality control using the Scalp Coupling Index (SCI) to identify and exclude bad channels.
-   Applies the Temporal Derivative Distribution Repair (TDDR) algorithm for motion correction.
-   Converts optical density to hemoglobin concentration using the modified Beer-Lambert Law.
-   Applies a band-pass filter (0.01 - 0.8 Hz) to isolate the signal of interest.
-   Calculates a channel-wise functional connectivity matrix (Pearson correlation) for the HbO signal.
-   Outputs for each subject:
    -   A `.csv` file containing the full connectivity matrix, with bad channels marked as `NA`.
    -   A `.png` heatmap visualization of the connectivity matrix.

### Usage

To run the connectivity pipeline, provide the path to the directory containing your `.snirf` files.

```bash
python fnirs_connectivity.py /path/to/your/data_directory
```

The script will create a new directory named `fnirs_connectivity_results` in the parent directory of your data folder. This new directory will contain the connectivity matrices and heatmap images.

---

## Installation

To use these scripts, you need to have Python 3 installed, along with several scientific computing packages.

1.  **Clone the repository (or download the files):**
    ```bash
    git clone https://github.com/hollabharath/fNIRS_PARAM
    cd fNIRS_PARAM
    ```

2.  **Install the required packages:**
    It is recommended to use a virtual environment. The dependencies are listed in the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

### Dependencies

-   `mne`
-   `mne_nirs`
-   `pandas`
-   `numpy`
-   `matplotlib`
-   `seaborn`


