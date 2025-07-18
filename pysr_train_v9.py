#!/usr/bin/env python3
"""
PySR High-Dimensional Data Analysis Runner

This script performs symbolic regression on high-dimensional datasets using PySR.
It supports multiple processing modes and can be configured via JSON files.

Processing Modes:
-----------------
1. separate_validation: Use separate training and validation dataset files
   - Requires: training_datasets and validation_datasets lists
   - Use when: You have pre-split train/validation data in separate files
   
2. single_dataset_split: Use a single dataset file and split it into train/val/test
   - Requires: training_datasets list (validation_datasets ignored)
   - Use when: You have one large dataset to split programmatically
   
3. batch_noise_folders: Process multiple noise folders, each containing CSV files
   - Requires: base_directory containing folders ending with 'Noise'
   - Use when: You have multiple experiment folders to process in batch

Usage:
------
python pysr_runner.py --config config.json
python pysr_runner.py --config config.json --mode separate_validation
python pysr_runner.py --config config.json --runtime 3600
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pysr import PySRRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
import concurrent.futures as cf

# Import your custom modules
from stat_plots import *
from calculate_errors import *
from dataset_plots import *

# MATPLOTLIB FONT SETTINGS
from matplotlib.font_manager import FontProperties
font_prop = FontProperties(size=26, family='serif', style='normal')
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({
    'font.size': 14,
    'lines.color': 'orange',
    'scatter.edgecolors': '#1F77B4'
})


class PySRRunner:
    """Main class for running PySR experiments with different configurations."""
    
    VALID_MODES = ['separate_validation', 'single_dataset_split', 'batch_noise_folders']
    
    def __init__(self, config_path):
        """Initialize runner with configuration file."""
        self.config = self.load_config(config_path)
        self.validate_config()
        
    def load_config(self, config_path):
        """Load configuration from JSON file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        return config
    
    def validate_config(self):
        """Validate configuration parameters."""
        runtime_params = self.config.get('runtime_params', {})
        mode = runtime_params.get('processing_mode', '')
        
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid processing_mode: {mode}. Must be one of {self.VALID_MODES}")
            
        # Validate mode-specific requirements
        if mode == 'separate_validation':
            if not runtime_params.get('training_datasets') or not runtime_params.get('validation_datasets'):
                raise ValueError("separate_validation mode requires both training_datasets and validation_datasets")
            if len(runtime_params['training_datasets']) != len(runtime_params['validation_datasets']):
                raise ValueError("Number of training and validation datasets must match")
                
        elif mode == 'single_dataset_split':
            if not runtime_params.get('training_datasets'):
                raise ValueError("single_dataset_split mode requires training_datasets")
                
        elif mode == 'batch_noise_folders':
            if not runtime_params.get('base_directory'):
                raise ValueError("batch_noise_folders mode requires base_directory")
    
    def run(self, sequential=False):
        """Execute the appropriate processing mode."""
        mode = self.config['runtime_params']['processing_mode']
        
        print(f"\n{'='*60}")
        print(f"Running PySR in {mode} mode")
        print(f"{'='*60}\n")
        
        if mode == 'separate_validation':
            self.run_separate_validation(sequential=sequential)
        elif mode == 'single_dataset_split':
            self.run_single_dataset_split()
        elif mode == 'batch_noise_folders':
            self.run_batch_noise_folders()
    
    def run_separate_validation(self, sequential=False, train_file=None, val_file=None):
        """Process datasets with separate training and validation files."""
        runtime_params = self.config['runtime_params']
        pysr_params = self.config['pysr_params']
        # If specific files are provided, process only that pair
        if train_file and val_file:
            print(f"Processing user-specified files: train={train_file}, val={val_file}")
            process_large_dataset_with_validation(
                runtime_params['train_directory'],
                train_file,
                runtime_params['val_directory'],
                val_file,
                runtime_params['runtime'],
                runtime_params['scale_boolean'],
                runtime_params['troubleshooting_boolean'],
                pysr_params
            )
            return
        if sequential:
            for train, val in zip(runtime_params['training_datasets'], runtime_params['validation_datasets']):
                print(f"Processing sequentially: train={train}, val={val}")
                process_large_dataset_with_validation(
                    runtime_params['train_directory'],
                    train,
                    runtime_params['val_directory'],
                    val,
                    runtime_params['runtime'],
                    runtime_params['scale_boolean'],
                    runtime_params['troubleshooting_boolean'],
                    pysr_params
                )
        else:
            parallel_process_datasets(
                runtime_params['training_datasets'],
                runtime_params['validation_datasets'],
                runtime_params['train_directory'],
                runtime_params['val_directory'],
                runtime_params['runtime'],
                runtime_params['scale_boolean'],
                runtime_params['troubleshooting_boolean'],
                pysr_params  # Pass PySR parameters from config
            )
    
    def run_single_dataset_split(self):
        """Process single datasets and split them into train/val/test."""
        runtime_params = self.config['runtime_params']
        pysr_params = self.config['pysr_params']
        
        for dataset in runtime_params['training_datasets']:
            print(f"Processing dataset: {dataset}")
            process_large_dataset(
                runtime_params['base_directory'],
                dataset,
                runtime_params['dataset_size'],
                runtime_params['runtime'],
                runtime_params['scale_boolean'],
                runtime_params['troubleshooting_boolean'],
                pysr_params  # Pass PySR parameters from config
            )
    
    def run_batch_noise_folders(self):
        """Process multiple folders ending with 'Noise'."""
        runtime_params = self.config['runtime_params']
        pysr_params = self.config['pysr_params']
        
        base_directory = runtime_params['base_directory']
        noise_folders = [f for f in os.listdir(base_directory) if f.endswith('Noise')]
        
        if not noise_folders:
            print(f"Warning: No folders ending with 'Noise' found in {base_directory}")
            return
            
        for folder in noise_folders:
            location = os.path.join(base_directory, folder)
            csv_files = [f for f in os.listdir(location) if f.endswith('.csv')]
            
            for file in csv_files:
                print(f'Processing - Folder: {folder}, File: {file}')
                process_large_dataset(
                    location,
                    file,
                    runtime_params['dataset_size'],
                    runtime_params['runtime'],
                    runtime_params['scale_boolean'],
                    runtime_params['troubleshooting_boolean'],
                    pysr_params  # Pass PySR parameters from config
                )


def main():
    """Main entry point for CLI execution."""
    parser = argparse.ArgumentParser(
        description='Run PySR symbolic regression on high-dimensional data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to JSON configuration file'
    )

    parser.add_argument(
        '--train_file', 
        type=str, 
        help='Specific training file'
    )
    
    parser.add_argument(
        '--val_file', 
        type=str, 
        help='Specific validation file'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=PySRRunner.VALID_MODES,
        help='Override processing mode from config'
    )
    
    parser.add_argument(
        '--runtime',
        type=int,
        help='Override runtime in seconds from config'
    )
    
    parser.add_argument(
        '--dataset-size',
        type=int,
        help='Override dataset size from config'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable troubleshooting mode'
    )
    
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Process separate_validation datasets sequentially instead of in parallel'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    runner = PySRRunner(args.config)
    
    # Apply command-line overrides
    if args.mode:
        runner.config['runtime_params']['processing_mode'] = args.mode
        runner.validate_config()  # Re-validate after override
        
    if args.runtime:
        runner.config['runtime_params']['runtime'] = args.runtime
        
    if args.dataset_size:
        runner.config['runtime_params']['dataset_size'] = args.dataset_size
        
    if args.debug:
        runner.config['runtime_params']['troubleshooting_boolean'] = True
    
    # Run the analysis
    try:
        # If both --train_file and --val_file are provided, run only that pair (sequentially, regardless of --sequential)
        if args.train_file and args.val_file:
            runner.run_separate_validation(sequential=True, train_file=args.train_file, val_file=args.val_file)
        else:
            runner.run(sequential=args.sequential)
    except Exception as e:
        print(f"Error during execution: {e}")
        sys.exit(1)


# ============================================================================
# CORE MODEL FUNCTIONS
# ============================================================================

def runModel(x_train, x_test, y_train, y_test, x_validate, y_validate, 
             func_name, directory, feature_columns, target_column, 
             runtime, scale_boolean, troubleshooting_boolean, pysr_params):
    """
    Trains a PySRRegressor model, evaluates, logs, and visualizes results.

    Args:
        x_train, x_test, y_train, y_test: Training and test data
        x_validate, y_validate: Validation data
        func_name: Name of the function/dataset
        directory: Directory to save results
        feature_columns: List of feature names
        target_column: Name of target column
        runtime: Time limit for PySRRegressor
        scale_boolean: Whether to scale data
        troubleshooting_boolean: Whether to print troubleshooting info
        pysr_params: Dictionary of PySR parameters from config

    Returns:
        None
    """
    # Change the working directory to the target directory
    os.chdir(directory)

    # Scale data if requested
    if scale_boolean:
        x_train, y_train, PredScaleFit_train, TargetScaleFit_train = scale_data_Nuttii(x_train, y_train)
        x_test, y_test, PredScaleFit_test, TargetScaleFit_test = scale_data_Nuttii(x_test, y_test)
        x_validate, y_validate, PredScaleFit_validate, TargetScaleFit_validate = scale_data_Nuttii(x_validate, y_validate)
    
    # Initialize PySR model with parameters from config
    params = pysr_params.copy()
    params["timeout_in_seconds"] = runtime
    model = PySRRegressor(**params)

    # Train the model
    start_time = datetime.now()
    print(f"Starting model training for {func_name}...")
    print(f"Training runtime = {np.mod(runtime/3600)} hours, {np.mod(runtime/60)} minutes, {runtime%60} seconds")
    model.fit(x_train, y_train)

    # Find best equation
    best_idx = _select_best_equation(model)
    
    # Prepare directory for saving results
    equation_dir = os.path.join(directory, "Info")
    os.makedirs(equation_dir, exist_ok=True)

    # After finding best_idx and if scale_boolean is True
    if scale_boolean:
        save_scaling_params_for_matlab(
            equation_dir, PredScaleFit_train, TargetScaleFit_train,
            feature_columns, model, best_idx
        )

    # Calculate metrics for best equation
    best_metrics = _calculate_best_equation_metrics(
        model, best_idx, x_train, y_train, x_test, y_test,
        scale_boolean, PredScaleFit_train if scale_boolean else None,
        TargetScaleFit_train if scale_boolean else None
    )
    
    # Save best equation info
    _save_best_equation_info(
        equation_dir, best_idx, model.sympy(best_idx), best_metrics,
        scale_boolean, PredScaleFit_train if scale_boolean else None,
        TargetScaleFit_train if scale_boolean else None
    )

    # Save all equations
    model.equations_.to_csv(os.path.join(equation_dir, "model_equations.csv"), index=False)

    # Process all equations and collect metrics
    all_metrics = _process_all_equations(
        model, x_train, x_test, x_validate, y_train, y_test, y_validate,
        scale_boolean, 
        (PredScaleFit_train, TargetScaleFit_train) if scale_boolean else (None, None),
        (PredScaleFit_test, TargetScaleFit_test) if scale_boolean else (None, None),
        (PredScaleFit_validate, TargetScaleFit_validate) if scale_boolean else (None, None),
        troubleshooting_boolean
    )

    # Create and save datalogs
    _create_and_save_datalogs(
        equation_dir, x_train, x_test, x_validate, y_train, y_test, y_validate,
        feature_columns, target_column, all_metrics
    )

    # Run linear regression comparison
    run_linear_regression_and_save(
        x_train, x_test, y_train, y_test, x_validate, y_validate,
        feature_columns, target_column, output_dir=equation_dir
    )

    # Create visualizations
    _create_visualizations(
        equation_dir, model, all_metrics, best_idx,
        x_train, y_train, y_test, y_validate, feature_columns
    )

    # Print completion time
    end_time = datetime.now()
    print(f"Total Duration: {(end_time - start_time).total_seconds()} seconds")


# ============================================================================
# DATASET PROCESSING FUNCTIONS
# ============================================================================

def process_large_dataset(base_directory, dataset, sample_size, runtime, 
                         scale_boolean, troubleshooting_boolean,pysr_params):
    """
    Loads a large dataset, splits into train/test/validation, and runs model pipeline.
    
    This function handles the 'single_dataset_split' mode where one dataset
    is programmatically split into train/test/validation sets.
    
    Args:
        base_directory: Directory containing the dataset
        dataset: Dataset filename
        sample_size: Number of rows for train/test/validation
        runtime: Time limit for PySRRegressor
        scale_boolean: Whether to scale data
        troubleshooting_boolean: Whether to print troubleshooting info
    
    Returns:
        None
    """
    print(f"Processing large dataset: {dataset}")

    # Load dataset
    df = pd.read_csv(os.path.join(base_directory, dataset))

    # Validate sample size
    if sample_size > len(df):
        raise ValueError(f"Sample size {sample_size} is larger than dataset size ({len(df)})")

    # Extract features and target
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)

    # Split data: first extract validation set, then split remaining into train/test
    X_train_all, X_val, y_train_all, y_val = train_test_split(
        x, y, test_size=sample_size, train_size=sample_size, random_state=42
    )
    
    # Further split training data into train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_all, y_train_all, 
        test_size=int(sample_size * 0.2), 
        train_size=int(sample_size * 0.8), 
        random_state=42
    )

    # Extract metadata
    func_name = dataset.split('.')[0]
    feature_columns = df.columns[:-1].tolist()
    target_column = df.columns[-1]

    # Create output directory
    dataset_dir = os.path.join(base_directory, func_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Run the model
    runModel(
        X_train, X_test, y_train, y_test, X_val, y_val, 
        func_name, dataset_dir, feature_columns, target_column,
        runtime, scale_boolean, troubleshooting_boolean, pysr_params
    )


def process_large_dataset_with_validation(training_directory, training_dataset, 
                                        validation_directory, validation_dataset, 
                                        runtime, scale_boolean, troubleshooting_boolean,pysr_params):
    """
    Loads separate training and validation datasets and runs model pipeline.
    
    This function handles the 'separate_validation' mode where training
    and validation data are in separate files.
    
    Args:
        training_directory: Directory for training data
        training_dataset: Training dataset filename
        validation_directory: Directory for validation data
        validation_dataset: Validation dataset filename
        runtime: Time limit for PySRRegressor
        scale_boolean: Whether to scale data
        troubleshooting_boolean: Whether to print troubleshooting info
    
    Returns:
        None
    """
    print(f"Processing datasets - Train: {training_dataset}, Val: {validation_dataset}")

    # Load datasets
    training_df = pd.read_csv(os.path.join(training_directory, training_dataset))
    validation_df = pd.read_csv(os.path.join(validation_directory, validation_dataset))

    # Warn about large datasets
    if len(training_df) > 1000:
        print("Warning: Training dataset exceeds 1000 rows, runtime may be slow.")

    # Extract training data
    x = training_df.iloc[:, :-1].values
    y = training_df.iloc[:, -1].values.reshape(-1, 1)

    # Extract validation data
    X_val = validation_df.iloc[:, :-1].values
    y_val = validation_df.iloc[:, -1].values.reshape(-1, 1)

    # Split training data into train/test (80/20)
    dataset_size = len(x)
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, 
        test_size=int(dataset_size * 0.2), 
        train_size=int(dataset_size * 0.8), 
        random_state=42
    )

    # Extract metadata
    func_name = training_dataset.split('.')[0]
    feature_columns = training_df.columns[:-1].tolist()
    target_column = training_df.columns[-1]

    # Create output directory
    dataset_dir = os.path.join(training_directory, func_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Run the model
    runModel(
        X_train, X_test, y_train, y_test, X_val, y_val,
        func_name, dataset_dir, feature_columns, target_column,
        runtime, scale_boolean, troubleshooting_boolean, pysr_params
    )


def parallel_process_datasets(train_datasets, val_datasets, train_dir, val_dir, 
                            runtime, scale_boolean, troubleshooting_boolean, pysr_params):
    """
    Parallelizes processing of multiple train/validation dataset pairs.
    
    Args:
        train_datasets: List of training dataset filenames
        val_datasets: List of validation dataset filenames
        train_dir: Directory containing training datasets
        val_dir: Directory containing validation datasets
        runtime: Time limit for each run
        scale_boolean: Whether to scale data
        troubleshooting_boolean: Whether to print troubleshooting info
    """
    print(f"Starting parallel processing of {len(train_datasets)} dataset pairs...")
    
    with cf.ProcessPoolExecutor() as executor:
        futures = []
        
        # Submit all dataset pairs for processing
        for train_dataset, val_dataset in zip(train_datasets, val_datasets):
            print(f"Submitting: train={train_dataset}, val={val_dataset}")
            futures.append(
                executor.submit(
                    process_large_dataset_with_validation,
                    train_dir, train_dataset, val_dir, val_dataset,
                    runtime, scale_boolean, troubleshooting_boolean, pysr_params
                )
            )
        
        # Wait for completion and handle errors
        completed = 0
        for future in cf.as_completed(futures):
            completed += 1
            try:
                future.result()
                print(f"Completed {completed}/{len(futures)} datasets")
            except Exception as exc:
                print(f"Dataset processing failed: {exc}")


# ============================================================================
# MODEL COMPARISON FUNCTIONS
# ============================================================================

def run_linear_regression_and_save(x_train, x_test, y_train, y_test, 
                                 x_validate, y_validate, feature_columns, 
                                 target_column, output_dir):
    """
    Fit a linear regression on the same data and save results for comparison.
    
    This provides a baseline to compare against symbolic regression results.
    
    Args:
        x_train, x_test, y_train, y_test, x_validate, y_validate: Data splits
        feature_columns: List of feature names
        target_column: Name of target column
        output_dir: Directory to save results
    """
    from sklearn.linear_model import LinearRegression
    from calculate_errors import calculate_mse_nmse, calculate_rmse

    print("Running linear regression for comparison...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Train linear regression
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    y_validate_pred = model.predict(x_validate)

    # Calculate metrics
    mse_train, nmse_train = calculate_mse_nmse(y_train, y_train_pred)
    mse_test, nmse_test = calculate_mse_nmse(y_test, y_test_pred)
    mse_validate, nmse_validate = calculate_mse_nmse(y_validate, y_validate_pred)
    
    rmse_train = calculate_rmse(y_train, y_train_pred)
    rmse_test = calculate_rmse(y_test, y_test_pred)
    rmse_validate = calculate_rmse(y_validate, y_validate_pred)

    # Save results
    with open(os.path.join(output_dir, "linear_regression_results.txt"), "w") as f:
        f.write("Linear Regression Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Train MSE: {mse_train:.6f}\n")
        f.write(f"Train NMSE: {nmse_train:.6f}\n")
        f.write(f"Train RMSE: {rmse_train:.6f}\n")
        f.write(f"Test MSE: {mse_test:.6f}\n")
        f.write(f"Test NMSE: {nmse_test:.6f}\n")
        f.write(f"Test RMSE: {rmse_test:.6f}\n")
        f.write(f"Validation MSE: {mse_validate:.6f}\n")
        f.write(f"Validation NMSE: {nmse_validate:.6f}\n")
        f.write(f"Validation RMSE: {rmse_validate:.6f}\n")
        f.write(f"\nCoefficients: {model.coef_}\n")
        f.write(f"Intercept: {model.intercept_}\n")

    # Save predictions
    vis_dir = os.path.join(output_dir, "Visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    pd.DataFrame(y_train_pred, columns=[f"{target_column}_pred"]).to_csv(
        os.path.join(vis_dir, "linear_train_pred.csv"), index=False
    )
    pd.DataFrame(y_test_pred, columns=[f"{target_column}_pred"]).to_csv(
        os.path.join(vis_dir, "linear_test_pred.csv"), index=False
    )
    pd.DataFrame(y_validate_pred, columns=[f"{target_column}_pred"]).to_csv(
        os.path.join(vis_dir, "linear_validate_pred.csv"), index=False
    )

    # Plot residuals
    _plot_linear_regression_residuals(
        y_train, y_train_pred, y_test, y_test_pred, vis_dir
    )


# ============================================================================
# UTILITY FUNCTIONS (Private)
# ============================================================================

def _select_best_equation(model):
    """Select the best equation from PySR results."""
    print("\nSelecting best equation...")
    print("Equations DataFrame:")
    print(model.equations_)

    # Validate equations dataframe
    if "loss" not in model.equations_ or "score" not in model.equations_:
        raise ValueError("Missing 'loss' or 'score' column in equations")

    # Filter based on loss
    min_loss = model.equations_["loss"].min()
    
    if min_loss > 0:
        # Filter to equations with loss < 2 * min_loss
        filtered_eqs = model.equations_.query(f"loss < {max(1e-9, 2 * min_loss)}")
    else:
        # If min loss is 0, use all equations
        print("Warning: Minimum loss is zero, using all equations")
        filtered_eqs = model.equations_

    # Select best equation by score
    if filtered_eqs.empty:
        print("Warning: No equations met filtering criteria, selecting best by score")
        best_idx = model.equations_["score"].idxmax()
    else:
        best_idx = filtered_eqs["score"].idxmax()

    print(f"Selected equation index: {best_idx}")
    print(f"Selected equation: {model.equations_.iloc[best_idx]}")
    
    return best_idx


def _calculate_best_equation_metrics(model, best_idx, x_train, y_train, 
                                   x_test, y_test, scale_boolean, 
                                   pred_scale_fit, target_scale_fit):
    """Calculate metrics for the best equation."""
    # Make predictions (these are in scaled space if scale_boolean=True)
    best_y_train_pred = model.predict(x_train, index=best_idx)
    best_y_test_pred = model.predict(x_test, index=best_idx)
    
    if scale_boolean:
        # Inverse transform predictions back to original scale
        best_y_train_pred = target_scale_fit.inverse_transform(
            best_y_train_pred.reshape(-1, 1)
        ).flatten()
        best_y_test_pred = target_scale_fit.inverse_transform(
            best_y_test_pred.reshape(-1, 1)
        ).flatten()
        
        # Also inverse transform the actual y values for comparison
        y_train_original = target_scale_fit.inverse_transform(y_train)
        y_test_original = target_scale_fit.inverse_transform(y_test)
    else:
        y_train_original = y_train
        y_test_original = y_test

    # NOW calculate metrics in original scale
    best_mse_train, best_nmse_train = calculate_mse_nmse(y_train_original, best_y_train_pred)
    best_mse_test, best_nmse_test = calculate_mse_nmse(y_test_original, best_y_test_pred)
    best_rmse_train = calculate_rmse(y_train_original, best_y_train_pred)
    best_rmse_test = calculate_rmse(y_test_original, best_y_test_pred)

    # Calculate residuals in original scale
    best_residuals_train = np.abs(y_train_original.flatten() - best_y_train_pred)
    best_residuals_test = np.abs(y_test_original.flatten() - best_y_test_pred)

    return {
        'mse_train': best_mse_train,
        'nmse_train': best_nmse_train,
        'rmse_train': best_rmse_train,
        'mse_test': best_mse_test,
        'nmse_test': best_nmse_test,
        'rmse_test': best_rmse_test,
        'residuals_train': best_residuals_train,
        'residuals_test': best_residuals_test
    }


def _save_best_equation_info(equation_dir, best_idx, best_eq, metrics,
                           scale_boolean, pred_scale_fit, target_scale_fit):
    """Save information about the best equation."""
    with open(os.path.join(equation_dir, "best.txt"), "a") as file:
        file.write(f"Selected Best Function: {best_idx}, {best_eq}\n")
        file.write(f"Train MSE: {metrics['mse_train']:.6f}\n")
        file.write(f"Train NMSE: {metrics['nmse_train']:.6f}\n")
        file.write(f"Train RMSE: {metrics['rmse_train']:.6f}\n")
        file.write(f"Test MSE: {metrics['mse_test']:.6f}\n")
        file.write(f"Test NMSE: {metrics['nmse_test']:.6f}\n")
        file.write(f"Test RMSE: {metrics['rmse_test']:.6f}\n")
        
        if scale_boolean and pred_scale_fit is not None:
            feature_means = pred_scale_fit.mean_
            feature_stds = pred_scale_fit.scale_
            target_mean = target_scale_fit.mean_
            target_std = target_scale_fit.scale_
            
            file.write("Feature Means: " + ", ".join([f"{m:.6f}" for m in feature_means]) + "\n")
            file.write("Feature Stds: " + ", ".join([f"{s:.6f}" for s in feature_stds]) + "\n")
            file.write(f"Target Mean: {target_mean[0]:.6f}\n")
            file.write(f"Target Std: {target_std[0]:.6f}\n")
        
        file.write("=" * 70 + "\n")


def _process_all_equations(model, x_train, x_test, x_validate, 
                         y_train, y_test, y_validate, scale_boolean,
                         train_scalers, test_scalers, validate_scalers,
                         troubleshooting_boolean):
    """Process all equations and calculate metrics."""
    # Initialize metric storage
    metrics = {
        'y_train_pred_list': [],
        'y_test_pred_list': [],
        'y_validation_pred_list': [],
        'mse_train_list': [],
        'nmse_train_list': [],
        'mse_test_list': [],
        'nmse_test_list': [],
        'mse_validation_list': [],
        'nmse_validation_list': [],
        'rmse_train_list': [],
        'rmse_test_list': [],
        'rmse_validation_list': [],
        'residuals_train_list': [],
        'residuals_test_list': [],
        'residuals_validation_list': []
    }

    # Inverse transform targets if scaled
    if scale_boolean:
        _, target_scale_train = train_scalers
        _, target_scale_test = test_scalers
        _, target_scale_validate = validate_scalers
        
        y_train = target_scale_train.inverse_transform(y_train)
        y_test = target_scale_test.inverse_transform(y_test)
        y_validate = target_scale_validate.inverse_transform(y_validate)

    # Process each equation
    for equation_idx in model.equations_.index:
        # Make predictions
        y_train_pred = model.predict(x_train, index=equation_idx)
        y_test_pred = model.predict(x_test, index=equation_idx)
        y_validation_pred = model.predict(x_validate, index=equation_idx)
        
        # Inverse transform predictions if scaled
        if scale_boolean:
            y_train_pred = target_scale_train.inverse_transform(
                y_train_pred.reshape(-1, 1)
            ).flatten()
            y_test_pred = target_scale_test.inverse_transform(
                y_test_pred.reshape(-1, 1)
            ).flatten()
            y_validation_pred = target_scale_validate.inverse_transform(
                y_validation_pred.reshape(-1, 1)
            ).flatten()

        # Calculate metrics
        mse_train, nmse_train = calculate_mse_nmse(y_train, y_train_pred)
        mse_test, nmse_test = calculate_mse_nmse(y_test, y_test_pred)
        mse_validation, nmse_validation = calculate_mse_nmse(y_validate, y_validation_pred)
        
        rmse_train = calculate_rmse(y_train, y_train_pred)
        rmse_test = calculate_rmse(y_test, y_test_pred)
        rmse_validation = calculate_rmse(y_validate, y_validation_pred)

        residuals_train = np.abs(y_train.flatten() - y_train_pred)
        residuals_test = np.abs(y_test.flatten() - y_test_pred)
        residuals_validation = np.abs(y_validate.flatten() - y_validation_pred)

        # Store metrics
        metrics['y_train_pred_list'].append(y_train_pred)
        metrics['y_test_pred_list'].append(y_test_pred)
        metrics['y_validation_pred_list'].append(y_validation_pred)
        metrics['mse_train_list'].append(mse_train)
        metrics['nmse_train_list'].append(nmse_train)
        metrics['mse_test_list'].append(mse_test)
        metrics['nmse_test_list'].append(nmse_test)
        metrics['mse_validation_list'].append(mse_validation)
        metrics['nmse_validation_list'].append(nmse_validation)
        metrics['rmse_train_list'].append(rmse_train)
        metrics['rmse_test_list'].append(rmse_test)
        metrics['rmse_validation_list'].append(rmse_validation)
        metrics['residuals_train_list'].append(residuals_train)
        metrics['residuals_test_list'].append(residuals_test)
        metrics['residuals_validation_list'].append(residuals_validation)

        if troubleshooting_boolean:
            print(f"\nEquation {equation_idx} shapes:")
            print(f"  y_train_pred: {y_train_pred.shape}")
            print(f"  residuals_train: {residuals_train.shape}")

    return metrics


def _create_and_save_datalogs(equation_dir, x_train, x_test, x_validate,
                            y_train, y_test, y_validate, feature_columns,
                            target_column, all_metrics):
    """Create and save comprehensive datalogs."""
    # Create primary datalogs with predictions and residuals
    train_datalog = pd.DataFrame(x_train, columns=feature_columns)
    train_datalog[target_column] = y_train
    
    test_datalog = pd.DataFrame(x_test, columns=feature_columns)
    test_datalog[target_column] = y_test
    
    validation_datalog = pd.DataFrame(x_validate, columns=feature_columns)
    validation_datalog[target_column] = y_validate

    # Add predictions and residuals for each model
    for i, preds in enumerate(all_metrics['y_train_pred_list']):
        train_datalog[f'Model_{i+1}_Y_pred'] = preds
        train_datalog[f'Model_{i+1}_Residuals'] = all_metrics['residuals_train_list'][i]

    for i, preds in enumerate(all_metrics['y_test_pred_list']):
        test_datalog[f'Model_{i+1}_Y_pred'] = preds
        test_datalog[f'Model_{i+1}_Residuals'] = all_metrics['residuals_test_list'][i]

    for i, preds in enumerate(all_metrics['y_validation_pred_list']):
        validation_datalog[f'Model_{i+1}_Y_pred'] = preds
        validation_datalog[f'Model_{i+1}_Residuals'] = all_metrics['residuals_validation_list'][i]

    # Save primary datalogs
    train_datalog.to_csv(os.path.join(equation_dir, "train_datalog_primary.csv"), index=False)
    test_datalog.to_csv(os.path.join(equation_dir, "test_datalog_primary.csv"), index=False)
    validation_datalog.to_csv(os.path.join(equation_dir, "validation_datalog_primary.csv"), index=False)

    # Create secondary datalogs with metrics
    metrics_datalog = pd.DataFrame({
        'Model': [f'Model_{i+1}' for i in range(len(all_metrics['mse_train_list']))],
        'MSE_Train': all_metrics['mse_train_list'],
        'NMSE_Train': all_metrics['nmse_train_list'],
        'RMSE_Train': all_metrics['rmse_train_list'],
        'MSE_Test': all_metrics['mse_test_list'],
        'NMSE_Test': all_metrics['nmse_test_list'],
        'RMSE_Test': all_metrics['rmse_test_list'],
        'MSE_Validation': all_metrics['mse_validation_list'],
        'NMSE_Validation': all_metrics['nmse_validation_list'],
        'RMSE_Validation': all_metrics['rmse_validation_list']
    })
    
    metrics_datalog.to_csv(os.path.join(equation_dir, "metrics_summary.csv"), index=False)

    # Save raw data for debugging
    raw_dir = os.path.join(equation_dir, "Raw Data")
    os.makedirs(raw_dir, exist_ok=True)
    
    for key, data in all_metrics.items():
        if isinstance(data, list) and len(data) > 0:
            pd.DataFrame(data).to_csv(
                os.path.join(raw_dir, f"{key}.csv"), index=False
            )


def _create_visualizations(equation_dir, model, all_metrics, best_idx,
                         x_train, y_train, y_test, y_validate, feature_columns):
    """Create all visualization plots."""
    print("\nCreating visualizations...")
    
    visualization_dir = os.path.join(equation_dir, "Visualizations")
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Main complexity vs error plots
    plot_complexity_vs_error(
        model.equations_['complexity'], 
        all_metrics['nmse_train_list'], 
        all_metrics['nmse_validation_list'], 
        save_path=visualization_dir, 
        best=best_idx
    )
    
    plot_complexity_vs_error_loglog(
        model.equations_['complexity'], 
        all_metrics['nmse_train_list'], 
        all_metrics['nmse_validation_list'], 
        save_path=visualization_dir,
        best=best_idx
    )
    
    # Compare with linear regression
    compare_regression_metrics(equation_dir, visualization_dir)
    
    # Variable density visualization
    plot_variable_density(
        x_train, feature_columns, 
        output_dir=visualization_dir, 
        filename_prefix="train_density"
    )
    
    # Training visualizations
    training_dir = os.path.join(visualization_dir, "Training")
    os.makedirs(training_dir, exist_ok=True)
    plot_residual_kde(all_metrics['residuals_train_list'], output_dir=training_dir)
    scatter_residuals_vs_true(all_metrics['residuals_train_list'], y_train, output_dir=training_dir)
    
    # Testing visualizations
    testing_dir = os.path.join(visualization_dir, "Testing")
    os.makedirs(testing_dir, exist_ok=True)
    plot_residual_kde(all_metrics['residuals_test_list'], output_dir=testing_dir)
    scatter_residuals_vs_true(all_metrics['residuals_test_list'], y_test, output_dir=testing_dir)
    
    # Validation visualizations
    validation_dir = os.path.join(visualization_dir, "Validation")
    os.makedirs(validation_dir, exist_ok=True)
    plot_residual_kde(all_metrics['residuals_validation_list'], output_dir=validation_dir)
    scatter_residuals_vs_true(all_metrics['residuals_validation_list'], y_validate, output_dir=validation_dir)
    
    print("Visualizations complete!")


def _plot_linear_regression_residuals(y_train, y_train_pred, y_test, y_test_pred, output_dir):
    """Plot residuals for linear regression baseline."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training residuals
    train_residuals = y_train.flatten() - y_train_pred.flatten()
    ax1.scatter(y_train, train_residuals, alpha=0.5)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Linear Regression Training Residuals')
    ax1.grid(True, alpha=0.3)
    
    # Test residuals
    test_residuals = y_test.flatten() - y_test_pred.flatten()
    ax2.scatter(y_test, test_residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Linear Regression Test Residuals')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'linear_regression_residuals.png'), dpi=150)
    plt.close()


# ============================================================================
# DATA SCALING FUNCTIONS
# ============================================================================

def scale_data_Nuttii(X_predictor, Y_target):
    """
    Scale predictor and target data using StandardScaler.
    
    This function scales both features and targets to have zero mean and unit variance,
    which can help with numerical stability in symbolic regression.
    
    Args:
        X_predictor: Feature matrix (n_samples, n_features)
        Y_target: Target vector (n_samples, 1)
    
    Returns:
        X_predictor_fit: Scaled features
        Y_target_fit: Scaled target
        PredScaleFit: Fitted StandardScaler for features
        TargetScaleFit: Fitted StandardScaler for target
    """
    # Initialize scalers
    PredScaleFit = StandardScaler()
    TargetScaleFit = StandardScaler()
    
    # Fit and transform features
    X_predictor_fit = PredScaleFit.fit_transform(X_predictor)
    
    # Fit and transform target (ensure 2D for scaler)
    Y_target_fit = TargetScaleFit.fit_transform(Y_target.reshape(-1, 1))
    
    return X_predictor_fit, Y_target_fit, PredScaleFit, TargetScaleFit


# ============================================================================
# COMPARISON VISUALIZATION FUNCTIONS
# ============================================================================

def compare_regression_metrics(info_dir, output_dir):
    """
    Compare symbolic regression results with linear regression baseline.
    
    Creates visualization comparing the best symbolic regression model
    with linear regression to show improvement.
    
    Args:
        info_dir: Directory containing result files
        output_dir: Directory to save visualization
    """
    # Check if linear regression results exist
    linear_results_path = os.path.join(info_dir, "linear_regression_results.txt")
    if not os.path.exists(linear_results_path):
        print("No linear regression results found for comparison")
        return
    
    # Parse linear regression results
    linear_metrics = {}
    with open(linear_results_path, 'r') as f:
        content = f.read()
        for line in content.split('\n'):
            if 'MSE:' in line or 'NMSE:' in line or 'RMSE:' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = float(parts[1].strip())
                    linear_metrics[key] = value
    
    # Parse symbolic regression results
    best_results_path = os.path.join(info_dir, "best.txt")
    if not os.path.exists(best_results_path):
        print("No symbolic regression results found for comparison")
        return
    
    symb_metrics = {}
    with open(best_results_path, 'r') as f:
        content = f.read()
        for line in content.split('\n'):
            if 'MSE:' in line or 'NMSE:' in line or 'RMSE:' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = float(parts[1].strip())
                    symb_metrics[key] = value
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['MSE', 'NMSE', 'RMSE']
    datasets = ['Train', 'Test', 'Validation']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        linear_values = []
        symb_values = []
        labels = []
        
        for dataset in datasets:
            key = f"{dataset} {metric}"
            if key in linear_metrics and key in symb_metrics:
                linear_values.append(linear_metrics[key])
                symb_values.append(symb_metrics[key])
                labels.append(dataset)
        
        if linear_values and symb_values:
            x = np.arange(len(labels))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, linear_values, width, label='Linear Regression')
            bars2 = ax.bar(x + width/2, symb_values, width, label='Symbolic Regression')
            
            ax.set_xlabel('Dataset')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.4f}',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3),
                              textcoords="offset points",
                              ha='center', va='bottom',
                              fontsize=8)
    
    plt.suptitle('Linear vs Symbolic Regression Performance', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regression_comparison.png'), dpi=150)
    plt.close()


def plot_variable_density(X, feature_names, output_dir, filename_prefix="density"):
    """
    Plot kernel density estimates for each input variable.
    
    Useful for understanding the distribution of input features.
    
    Args:
        X: Feature matrix
        feature_names: List of feature names
        output_dir: Directory to save plots
        filename_prefix: Prefix for saved files
    """
    from scipy import stats
    
    n_features = X.shape[1]
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, (feature, ax) in enumerate(zip(feature_names, axes)):
        data = X[:, i]
        
        # Calculate KDE
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        density = kde(x_range)
        
        # Plot
        ax.plot(x_range, density, 'b-', linewidth=2)
        ax.fill_between(x_range, density, alpha=0.3)
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution of {feature}')
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename_prefix}.png'), dpi=150)
    plt.close()


def save_scaling_params_for_matlab(equation_dir, pred_scale_fit, target_scale_fit, 
                                  feature_columns, model, best_idx):
    """Save scaling parameters in a format MATLAB can easily use."""
    import scipy.io
    
    # Prepare scaling data
    scaling_data = {
        'feature_names': feature_columns,
        'feature_means': pred_scale_fit.mean_,
        'feature_stds': pred_scale_fit.scale_,
        'target_mean': target_scale_fit.mean_[0],
        'target_std': target_scale_fit.scale_[0],
        'equation_string': str(model.sympy(best_idx)),
        'scaled_equation': str(model.sympy(best_idx))  # This is in scaled space
    }
    
    # Save as .mat file
    scipy.io.savemat(os.path.join(equation_dir, 'scaling_params.mat'), scaling_data)
    
    # Also create a MATLAB function template
    with open(os.path.join(equation_dir, 'apply_sr_equation.m'), 'w') as f:
        f.write("function output = apply_sr_equation(input_values, equation_func)\n")
        f.write("    % Load scaling parameters\n")
        f.write("    load('scaling_params.mat');\n")
        f.write("    \n")
        f.write("    % Scale inputs\n")
        f.write("    scaled_inputs = (input_values - feature_means) ./ feature_stds;\n")
        f.write("    \n")
        f.write("    % Apply equation (in scaled space)\n")
        f.write("    scaled_output = equation_func(scaled_inputs);\n")
        f.write("    \n")
        f.write("    % Unscale output\n")
        f.write("    output = scaled_output * target_std + target_mean;\n")
        f.write("end\n")




if __name__ == "__main__":
    main()