# Numerical Stability Analysis Repository

This repository contains a Python script (`deepseek-script.py`) for analyzing the numerical stability, focusing on the consistency of token probabilities and rankings between different runs or environments.  It loads data from JSON files, calculates various metrics (epsilon, perplexity, ranking preservation), and generates visualizations.

## Table of Contents

1.  [Overview](#overview)
2.  [File Structure](#file-structure)
3.  [Dependencies](#dependencies)
4.  [Usage](#usage)
    *   [Configuration](#configuration)
    *   [Running the Script](#running-the-script)


## 1. Overview

The script compares "original" data (presumably from a stable, reference environment) with "check" data (from a potentially less stable or different environment).  The core idea is to quantify how much the model's output probabilities and token rankings change between these two sets of data.  This helps assess the robustness of the model to numerical variations.  The script calculates and visualizes several key stability metrics.

## 2. File Structure

The repository is expected to have the following structure:

```
di-verification/
├── db_1/             <-- Directory containing original data files (JSON)
├── db_2/             <-- Directory containing original data files (JSON)
├── db_3/             <-- Directory containing original data files (JSON)
├── db_4/             <-- Directory containing original data files (JSON)
├── db_5/             <-- Directory containing original data files (JSON)
├── db_13/            <-- Directory containing data to be checked (JSON)
├── analysis_results/ <-- Output directory for plots and summary
└── deepseek-script.py  <-- The main Python script
└── README.md           <-- This file
```

*   **`db_1/`, `db_2/`, ..., `db_5/`**:  Directories containing the "original" JSON data files.  Each file represents a single data instance (e.g., a sequence or a batch).
*   **`db_13/`**: The directory containing "check" JSON data files.  These files are named in a specific way to link them to their corresponding original files (explained later).
*   **`analysis_results/`**: This directory will be created by the script (if it doesn't exist) to store the generated plots and a summary of the results.
*   **`deepseek-script.py`**: The Python script performing the analysis.

**Expected JSON format:** The script assumes that the data files within `db_x` and `db_13` directories are in JSON format with a particular structure. Here is a detailed breakdown with an example.

*   **Original data file (`db_1` to `db_5`)**

```json
{
  "execution_data": [
    {
      "id": 123,
      "prob": 0.85,
      "top_k": [
        {"id": 456, "prob": 0.7},
        {"id": 789, "prob": 0.2},
        {"id": 101, "prob": 0.05}
      ]
    },
    {
      "id": 234,
      "prob": 0.92,
      "top_k": [
        {"id": 567, "prob": 0.8},
        {"id": 890, "prob": 0.1},
        {"id": 112, "prob": 0.02}
      ]
    }
  ]
}
```

*   `execution_data`: A list of dictionaries, each representing a "step" or "token" in the sequence.
    *   `id`: The ID of the predicted token at this step.
    *   `prob`: The probability assigned to the predicted token (`id`).
    *   `top_k`: A list of the top-K tokens and their probabilities.  Each entry is a dictionary:
        *   `id`: The ID of a token in the top-K.
        *   `prob`: The probability of that token.

*   **Check data file (`db_13`)**

```json
{
  "check_data": [
    {
      "id": 123,
      "prob": 0.84,
      "top_k": [
        {"id": 456, "prob": 0.68},
        {"id": 789, "prob": 0.21},
        {"id": 101, "prob": 0.06}
      ]
    },
    {
      "id": 234,
      "prob": 0.93,
      "top_k": [
        {"id": 567, "prob": 0.81},
        {"id": 890, "prob": 0.09},
        {"id": 212, "prob": 0.03}
      ]
    }
  ]
}
```
The structure is *identical* to the `execution_data` in the original files, but it's named `check_data` at the top level. It is crucial that corresponding entries between `execution_data` and `check_data` have the same length and represent the same token positions.

## 3. Dependencies

The script requires the following Python libraries:

*   **`json`**: For loading data from JSON files. (Usually included with Python)
*   **`os`**: For file system operations. (Usually included with Python)
*   **`numpy`**: For numerical operations (e.g., calculating perplexity).
*   **`matplotlib`**: For generating plots.
*   **`tqdm`**: For displaying progress bars.
*   **`collections`**: For using `defaultdict`. (Usually included with Python)

You can install these dependencies using `pip`:

```bash
pip install numpy matplotlib tqdm
```

## 4. Usage

### 4.1 Configuration

Before running the script, you might need to adjust the following configuration parameters at the beginning of `deepseek-script.py`:

*   **`BASE_DIRS`**:  A list of directories containing the "original" data files.  Modify this if you have your original data in different locations.
*   **`CHECK_DIR`**: The directory containing the "check" data files.  Change this to point to your check data.
*   **`K`**:  The number of top tokens to consider for ranking preservation (Top-K preservation rate).
*   **`M`**:  Used in the margin calculation (examines the top K+M tokens).
*   **`RESULTS_DIR`**: The directory where the results (plots and summary) will be saved.
* **`CUT_OFF_TOP_P`**: The threshold value to determine whether the margin calculation should be applied.

### 4.2 Running the Script

1.  **Place your data files:** Ensure your "original" JSON files are in the directories specified by `BASE_DIRS`, and your "check" JSON files are in `CHECK_DIR`. The "check" files should be named `check_machine_<original_filename>.json`.  For example, if you have an original file `db_1/data_001.json`, the corresponding check file should be named `db_13/check_machine_data_001.json`. The script uses this naming convention to match original and check files.

2.  **Navigate to the directory:** Open a terminal or command prompt and navigate to the directory containing the `deepseek-script.py` script.

3.  **Run the script:** Execute the script using Python:

    ```bash
    python deepseek-script.py
    ```

