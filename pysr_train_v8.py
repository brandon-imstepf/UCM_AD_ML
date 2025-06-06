import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from pysr import PySRRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
import tkinter as tk
from tkinter import messagebox
from stat_plots import *
from calculate_errors import *
import concurrent.futures as cf

# MATPLOTLIB FONT SETTINGS #
from matplotlib.font_manager import FontProperties
font_prop = FontProperties(size=26, family='serif', style='normal')
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({
    'font.size': 14,
    'lines.color': 'orange',
    'scatter.edgecolors': '#1F77B4'
})
# ------------------------ #





''' // For v8 // 
Need to:
    - Record mean, SD for z-score transformation ++
    + add RMSE ++
    + Add a variable density visualization ++
'''

# -------------------- RUNTIME PARAMS -------------------- #
# All runtime parameters for easy configuration
RUNTIME_PARAMS = {
    'base_directory': r'C:\Users\brand\Desktop\Raj-Sindi\training_data\sim_csv_v11\train',
    'train_directory': r'C:\Users\brand\Desktop\Raj-Sindi\training_data\sim_csv_v11\train',
    'val_directory': r'C:\Users\brand\Desktop\Raj-Sindi\training_data\sim_csv_v11\val',
    'training_datasets': [
        'flux_only_nobias_train_e3.csv',
        'flux_only_bias_train_e3.csv',
        'w1_only_nobias_train_e3.csv',
        'w1_only_bias_train_e3.csv'
    ],
    'validation_datasets': [
        'flux_only_nobias_val_e3.csv',
        'flux_only_bias_val_e3.csv',
        'w1_only_nobias_val_e3.csv',
        'w1_only_bias_val_e3.csv'
    ],
    'dataset_size': 1000,  # test/train and validate size
    'runtime': 60*5,       # in seconds
    'scale_boolean': True,
    'troubleshooting_boolean': False,
    'batch_boolean': False,
    'val_batch_boolean': True, # If you want to use process_large_dataset() which means you have a single dataset for training and validation
    'separate_validation': True # If you want to use process_large_dataset_with_validation() which means you have separate validation datasets 
}
# -------------------------------------------------------- #

global scale_boolean, troubleshooting_boolean

# Unpack runtime params for legacy code compatibility
base_directory = RUNTIME_PARAMS['base_directory']
train_directory = RUNTIME_PARAMS['train_directory']
val_directory = RUNTIME_PARAMS['val_directory']
training_datasets = RUNTIME_PARAMS['training_datasets']
validation_datasets = RUNTIME_PARAMS['validation_datasets']
dataset_size = RUNTIME_PARAMS['dataset_size']
runtime = RUNTIME_PARAMS['runtime']
scale_boolean = RUNTIME_PARAMS['scale_boolean']
troubleshooting_boolean = RUNTIME_PARAMS['troubleshooting_boolean']
batch_boolean = RUNTIME_PARAMS['batch_boolean']
val_batch_boolean = RUNTIME_PARAMS['val_batch_boolean']
separate_validation = RUNTIME_PARAMS['separate_validation']




# -------------------- PYSR PARAMS -------------------- #
PYSR_PARAMS = {
    "procs": 4,
    "populations": 32,
    "population_size": 500,
    "ncycles_per_iteration": 10000,
    "niterations": 100000,
    "complexity_of_constants": 5,
    "constraints": {
        "/": (-1, 9),
        "^": (-1, 1),
        "square": 9,
        "cube": 9,
        "exp": 9,
    },
    "early_stop_condition": "stop_if(loss,complexity) = loss < 1e-7 && complexity < 20 || loss < 1e-9",
    "timeout_in_seconds": runtime,
    "maxsize": 45,
    "maxdepth": 5,
    "binary_operators": ["*", "+", "-", "/", "^"],
    "unary_operators": ["exp", "log"],
    "nested_constraints": {
        "log": {"log": 0},
        "/": {"log": 0},
    },
    "progress": False,
    "weight_randomize": 5,
    "weight_add_node": 3,
    "weight_optimize": 0.001,
    "precision": 32,
    "warm_start": False,
    "turbo": False,
    "denoise": False,
    "model_selection": "accurate",
}
# ----------------------------------------------------- #

def runModel(x_train, x_test, y_train, y_test, x_validate, y_validate, func_name, directory, feature_columns, target_column, runtime, scale_boolean, troubleshooting_boolean):
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

    Returns:
        None
    """
    # Change the working directory to the target directory
    os.chdir(directory)

    if scale_boolean:
        # scale data to make it easier
        #x, y = scale_data_ZScore(x, y)
        x_train,y_train,PredScaleFit_train,TargetScaleFit_train = scale_data_Nuttii(x_train,y_train)
        x_test,y_test,PredScaleFit_test,TargetScaleFit_test = scale_data_Nuttii(x_test,y_test)
        x_validate, y_validate, PredScaleFit_validate,TargetScaleFit_validate = scale_data_Nuttii(x_validate,y_validate)
        
    
    # Learn equations using PySR
    params = PYSR_PARAMS.copy()
    params["timeout_in_seconds"] = runtime  # ensure runtime is current
    model = PySRRegressor(**params)

    start_time = datetime.now()

    model.fit(x_train, y_train)

    global best_idx

    #Print equations for debugging
    print("Equations DataFrame:")
    print(model.equations_)

    # Check if "loss" and "score" exist
    if "loss" not in model.equations_ or "score" not in model.equations_:
        print("Error: Missing 'loss' or 'score' column in equations.")
        return None

    # Ensure loss.min() is valid
    min_loss = model.equations_["loss"].min()
    if min_loss == 0:
        print("Warning: Minimum loss is zero. Adjusting filtering strategy.")

    # Apply filtering only if min_loss > 0
    if min_loss > 0:
        filtered_eqs = model.equations_.query(f"loss < {max(1e-9, 2 * min_loss)}")  # Avoid loss < 0 issues
    else:
        filtered_eqs = model.equations_  # Use all equations if loss is zero

    # If filtering returns empty, select based on score
    if filtered_eqs.empty:
        print("Warning: No equations met the loss filtering criteria. Selecting the best available equation instead.")
        best_idx = model.equations_["score"].idxmax()  # Pick the best equation based on score
    else:
        best_idx = filtered_eqs["score"].idxmax()  # Select from filtered equations

    # Print selected equation
    print(f"Best equation index: {best_idx}")
    print("Selected equation:", model.equations_.iloc[best_idx])

    # Prepare directory for saving results
    equation_dir = os.path.join(directory, "Info")
    os.makedirs(equation_dir, exist_ok=True)

    # --- Linear Regression Comparison --- #
    from dataset_plots import plot_linear_regression_residuals, compare_regression_metrics
    run_linear_regression_and_compare(
        x_train, x_test, y_train, y_test, x_validate, y_validate,
        feature_columns, target_column, output_dir=equation_dir
    )
    # Move linear regression residual plot to Visualizations
    plot_linear_regression_residuals(
        y_train, y_train_pred=None, y_test=y_test, y_test_pred=None,  # Will be loaded inside the function
        output_dir=os.path.join(equation_dir, "Visualizations")
    )
    # Compare symbolic and linear regression metrics visually
    compare_regression_metrics(
        equation_dir, os.path.join(equation_dir, "Visualizations")
    )
    # for BEST equation #
    best_eq = model.sympy(best_idx)
    best_y_train_pred = model.predict(x_train, index=best_idx)
    best_y_test_pred = model.predict(x_test, index=best_idx)

    best_mse_train, best_nmse_train = calculate_mse_nmse(y_train, best_y_train_pred)
    best_mse_test, best_nmse_test = calculate_mse_nmse(y_test,best_y_test_pred)
    best_rmse_train = calculate_rmse(y_train, best_y_train_pred)
    best_rmse_test = calculate_rmse(y_test, best_y_test_pred)

    best_residuals_train = np.abs(y_train.flatten() - best_y_train_pred)
    best_residuals_test = np.abs(y_test.flatten() - best_y_test_pred)
    # ----------------------------------------------------------- #

    # Record mean and SD for z-score transformation
    if scale_boolean:
        feature_means = PredScaleFit_train.mean_
        feature_stds = PredScaleFit_train.scale_
        target_mean = TargetScaleFit_train.mean_
        target_std = TargetScaleFit_train.scale_
    else:
        feature_means = feature_stds = target_mean = target_std = None

    # Write equation, timing, mean/SD, and RMSE to log
    with open(os.path.join(equation_dir, "best.txt"), "a") as file:
        file.write(f"Selected Best Function: {best_idx}, {best_eq}\n")
        file.write(f"Train MSE: {best_mse_train}\n")
        file.write(f"Train NMSE: {best_nmse_train}\n")
        file.write(f"Train RMSE: {best_rmse_train}\n")
        file.write(f"Test MSE: {best_mse_test}\n")
        file.write(f"Test NMSE: {best_nmse_test}\n")
        file.write(f"Test RMSE: {best_rmse_test}\n")
        if scale_boolean:
            file.write("Feature Means: " + ", ".join([f"{m:.6f}" for m in feature_means]) + "\n")
            file.write("Feature Stds: " + ", ".join([f"{s:.6f}" for s in feature_stds]) + "\n")
            file.write(f"Target Mean: {target_mean[0]:.6f}\n")
            file.write(f"Target Std: {target_std[0]:.6f}\n")
        file.write("***************************************************************\n")
        file.write(f"Train Residuals: {best_residuals_train}\n")
        file.write("***************************************************************\n")
        file.write(f"Test Residuals: {best_residuals_test}\n")
        file.write("***************************************************************\n")
    # ----------------------------------------------------------- #
    

    # Additional troubleshooting file
    model.equations_.to_csv(os.path.join(equation_dir, "model_equations.csv"), index=False)

    # Initialize lists to store all log values
    y_validation_pred_list = []
    y_train_pred_list = []
    y_test_pred_list = []
    mse_train_list, nmse_train_list = [],[]
    mse_test_list, nmse_test_list = [],[]
    residuals_train_list = []
    residuals_test_list = []
    mse_validation_list = []
    nmse_validation_list = []
    residuals_validation_list = []
    rmse_train_list, rmse_test_list, rmse_validation_list = [],[],[]

    # Inverse Transform the targets
    if scale_boolean:
        y_train = TargetScaleFit_train.inverse_transform(y_train)
        y_test = TargetScaleFit_test.inverse_transform(y_test)
        y_validate = TargetScaleFit_validate.inverse_transform(y_validate)

    for equation_idx in model.equations_.index:
        y_train_pred = model.predict(x_train, index=equation_idx)
        y_test_pred = model.predict(x_test, index=equation_idx)
        y_validation_pred = model.predict(x_validate, index=equation_idx)
        
        if scale_boolean:
            print(f'The shape of y_train_pred before reshaping is: {np.shape(y_train_pred)}') # should be (800,)
            print(f'The shape of y_test_pred before reshaping is: {np.shape(y_test_pred)}') # should be (200,)
            print(f'The shape of y_validation_pred before reshaping is: {np.shape(y_validation_pred)}') # should be (1000,)
            y_train_pred = y_train_pred.reshape(-1, 1)  # should be (800,)
            #print(f'The shape of y_train_pred AFTER reshaping is: {np.shape(y_train_pred)}')
            # inverse_trasnform NEEDS to be 2D, it cannot be 1D.
            y_train_pred = TargetScaleFit_train.inverse_transform(y_train_pred).flatten()  # should be (800,1)
            print(f'The shape of y_train_pred before reshaping the transformation is: {np.shape(y_train_pred)}')
            
            y_test_pred = y_test_pred.reshape(-1, 1)  # should be (200,)
            #print(f'The shape of y_test_pred AFTER reshaping is: {np.shape(y_test_pred)}')
            y_test_pred = TargetScaleFit_test.inverse_transform(y_test_pred).flatten()  # should be (200,)

            y_validation_pred = y_validation_pred.reshape(-1, 1)  # should be (1000,)
            #print(f'The shape of y_validation_pred AFTER reshaping is: {np.shape(y_validation_pred)}')
            y_validation_pred = TargetScaleFit_validate.inverse_transform(y_validation_pred).flatten()  # should be (1000,)

        mse_train, nmse_train = calculate_mse_nmse(y_train, y_train_pred)
        mse_test, nmse_test = calculate_mse_nmse(y_test, y_test_pred)
        mse_validation, nmse_validation = calculate_mse_nmse(y_validate, y_validation_pred)
        
        rmse_train = calculate_rmse(y_train,y_train_pred)
        rmse_test = calculate_rmse(y_test,y_test_pred)
        rmse_validation = calculate_rmse(y_validate,y_validation_pred)

        residuals_train = np.abs(y_train.flatten() - y_train_pred)
        residuals_test = np.abs(y_test.flatten() - y_test_pred)
        residuals_validation = np.abs(y_validate.flatten() - y_validation_pred)        


        if troubleshooting_boolean:
            print(f"This is the output from equation {equation_idx}")
            print(f"The shape of y_train_pred is: {np.shape(y_train_pred)}")
            print(f"The shape of mse_train is: {np.shape(mse_train)}")
            print(f"The shape of nmse_train is: {np.shape(nmse_train)}")
            print(f"The shape of residuals_train is: {np.shape(residuals_train)}")
            print(f"The shape of y_test_pred is: {np.shape(y_test_pred)}")
            print(f"The shape of mse_test is: {np.shape(mse_test)}")
            print(f"The shape of nmse_test is: {np.shape(nmse_test)}")
            print(f"The shape of residuals_test is: {np.shape(residuals_test)}")
            print(f"The shape of y_validation_pred is: {np.shape(y_validation_pred)}")
            print(f"The shape of mse_validation is: {np.shape(mse_validation)}")
            print(f"The shape of nmse_validation is: {np.shape(nmse_validation)}")
            print(f"The shape of residuals_validations is: {np.shape(residuals_validation)}")
            print("***********************************************")
        
        # Append metrics to the respective lists

        y_validation_pred_list.append(y_validation_pred)
        y_train_pred_list.append(y_train_pred)
        y_test_pred_list.append(y_test_pred)
        mse_train_list.append(mse_train) 
        nmse_train_list.append(nmse_train)
        mse_test_list.append(mse_test) 
        nmse_test_list.append(nmse_test)
        residuals_train_list.append(residuals_train)
        residuals_test_list.append(residuals_test)
        mse_validation_list.append(mse_validation)
        nmse_validation_list.append(nmse_validation)
        residuals_validation_list.append(residuals_validation)
        rmse_train_list.append(rmse_train)
        rmse_test_list.append(rmse_test)
        rmse_validation_list.append(rmse_validation)

    # ----------------------------------------------------------- #


    print("***********************************************")
    print("***********************************************")
    print("***********************************************")
    print("This is the ouput of the shapes of the lists: ")
    print(f"The shape of y_train_pred_list is: {np.shape(y_train_pred_list)}")
    print(f"The shape of mse_train_list is: {np.shape(mse_train_list)}")
    print(f"The shape of nmse_train_list is: {np.shape(nmse_train_list)}")
    print(f"The shape of residuals_train_list is: {np.shape(residuals_train_list)}")
    print(f"The shape of y_test_pred_list is: {np.shape(y_test_pred_list)}")
    print(f"The shape of mse_test_list is: {np.shape(mse_test_list)}")
    print(f"The shape of nmse_test_list is: {np.shape(nmse_test_list)}")
    print(f"The shape of residuals_test_list is: {np.shape(residuals_test_list)}")
    print(f"The shape of y_validation_pred_list is: {np.shape(y_validation_pred_list)}")
    print(f"The shape of mse_validation_list is: {np.shape(mse_validation_list)}")
    print(f"The shape of nmse_validation_list is: {np.shape(nmse_validation_list)}")
    print(f"The shape of residuals_validations_list is: {np.shape(residuals_validation_list)}")
    print(f"The shape of rmse_train_list is: {np.shape(rmse_train_list)}")
    print(f"The shape of rmse_test_list is: {np.shape(rmse_test_list)}")
    print(f"The shape of rmse_validation_list is: {np.shape(rmse_validation_list)}")
    print("***********************************************")

    # Primary datalogs
    train_datalog_primary = pd.DataFrame(x_train, columns=feature_columns)
    train_datalog_primary[target_column] = y_train

    test_datalog_primary = pd.DataFrame(x_test, columns=feature_columns)
    test_datalog_primary[target_column] = y_test

    validation_datalog_primary = pd.DataFrame(x_validate, columns=feature_columns)
    validation_datalog_primary[target_column] = y_validate
    
    # --------------------------------------- #

    # populate primary datalogs #
    for i, preds in enumerate(y_train_pred_list):
        train_datalog_primary[f'Model_{i+1}_Y_pred'] = preds
    for i, residuals in enumerate(residuals_train_list):
        train_datalog_primary[f'Model_{i+1}_Residuals'] = residuals

    for i, preds in enumerate(y_test_pred_list):
        test_datalog_primary[f'Model_{i+1}_Y_pred'] = preds
    for i, residuals in enumerate(residuals_test_list):
        test_datalog_primary[f'Model_{i+1}_Residuals'] = residuals

    for i, preds in enumerate(y_validation_pred_list):
        validation_datalog_primary[f'Model_{i+1}_Y_pred'] = preds
    for i, residuals in enumerate(residuals_validation_list):
        validation_datalog_primary[f'Model_{i+1}_Residuals'] = residuals
    # --------------------------------------- #
    # populate secondary datalogs #
    train_datalog_secondary = pd.DataFrame({
        'Model': [f'Model_{i+1}' for i in range(len(mse_train_list))],
        'MSE_Train': mse_train_list,
        'NMSE_Train': nmse_train_list,
    })

    test_datalog_secondary = pd.DataFrame({
        'Model': [f'Model_{i+1}' for i in range(len(mse_test_list))],
        'MSE_Test': mse_test_list,
        'NMSE_Test': nmse_test_list,
    })

    validation_datalog_secondary = pd.DataFrame({
        'Model': [f'Model_{i+1}' for i in range(len(mse_validation_list))],
        'MSE_Validation': mse_validation_list,
        'NMSE_Validation': nmse_validation_list,
    })


    # --------------------------------------- #
    # Write datalogs to CSV #
    train_datalog_primary.to_csv(os.path.join(equation_dir, "train_datalog_primary.csv"), index=False)
    test_datalog_primary.to_csv(os.path.join(equation_dir, "test_datalog_primary.csv"), index=False)
    validation_datalog_primary.to_csv(os.path.join(equation_dir, "validation_datalog_primary.csv"), index=False)
    train_datalog_secondary.to_csv(os.path.join(equation_dir, "train_datalog_secondary.csv"), index=False)
    test_datalog_secondary.to_csv(os.path.join(equation_dir, "test_datalog_secondary.csv"), index=False)
    validation_datalog_secondary.to_csv(os.path.join(equation_dir, "validation_datalog_secondary.csv"), index=False)
    # Raw datalogs for debugging or further augmentation
    raw_directory = os.path.join(equation_dir, "Raw Data")
    os.makedirs(raw_directory, exist_ok=True)

    y_validation_pred_list_df = pd.DataFrame(y_validation_pred_list)
    y_validation_pred_list_df.to_csv(os.path.join(raw_directory, "y_validation_pred_list.csv"), index=False)

    y_train_pred_list_df = pd.DataFrame(y_train_pred_list)
    y_train_pred_list_df.to_csv(os.path.join(raw_directory, "y_train_pred_list.csv"), index=False)

    y_test_pred_list_df = pd.DataFrame(y_test_pred_list)
    y_test_pred_list_df.to_csv(os.path.join(raw_directory, "y_test_pred_list.csv"), index=False)

    mse_train_list_df = pd.DataFrame(mse_train_list)
    mse_train_list_df.to_csv(os.path.join(raw_directory, "mse_train_list.csv"), index=False)

    nmse_train_list_df = pd.DataFrame(nmse_train_list)
    nmse_train_list_df.to_csv(os.path.join(raw_directory, "nmse_train_list.csv"), index=False)

    mse_test_list_df = pd.DataFrame(mse_test_list)
    mse_test_list_df.to_csv(os.path.join(raw_directory, "mse_test_list.csv"), index=False)

    nmse_test_list_df = pd.DataFrame(nmse_test_list)
    nmse_test_list_df.to_csv(os.path.join(raw_directory, "nmse_test_list.csv"), index=False)

    residuals_train_list_df = pd.DataFrame(residuals_train_list)
    residuals_train_list_df.to_csv(os.path.join(raw_directory, "residuals_train_list.csv"), index=False)

    residuals_test_list_df = pd.DataFrame(residuals_test_list)
    residuals_test_list_df.to_csv(os.path.join(raw_directory, "residuals_test_list.csv"), index=False)

    mse_validation_list_df = pd.DataFrame(mse_validation_list)
    mse_validation_list_df.to_csv(os.path.join(raw_directory, "mse_validation_list.csv"), index=False)

    nmse_validation_list_df = pd.DataFrame(nmse_validation_list)
    nmse_validation_list_df.to_csv(os.path.join(raw_directory, "nmse_validation_list.csv"), index=False)

    residuals_validation_list_df = pd.DataFrame(residuals_validation_list)
    residuals_validation_list_df.to_csv(os.path.join(raw_directory, "residuals_validation_list.csv"), index=False)


    # --------------------------------------- #

    # Now to do stuff with the datalog ... #
    print("\n *********************************************** \n STARTING VISUALIZATIONS \n ***********************************************")
    visualization_dir = os.path.join(equation_dir, "Visualizations")
    os.makedirs(visualization_dir, exist_ok=True)
    # Main Graphs #
    plot_complexity_vs_error(model.equations_['complexity'], nmse_train_list, nmse_validation_list, save_path=visualization_dir, best=best_idx)
    plot_complexity_vs_error_loglog(model.equations_['complexity'], nmse_train_list, nmse_validation_list, save_path=visualization_dir,best=best_idx)
    # Variable density visualization for features
    from dataset_plots import plot_variable_density
    plot_variable_density(x_train, feature_columns, output_dir=visualization_dir, filename_prefix="train_density")
    # --------------------------------------- #
    # Training Graphs #
    training_dir = os.path.join(visualization_dir, "Training")
    plot_residual_kde(residuals_train_list, output_dir=training_dir)
    scatter_residuals_vs_true(residuals_train_list, y_train, output_dir=training_dir)
    #pca_residuals_visualization(train_datalog_primary, feature_columns, output_dir=training_dir)
    #parallel_coordinates_residuals(train_datalog_primary, feature_columns, output_dir=training_dir)
    # --------------------------------------- #
    # Testing Graphs #
    testing_dir = os.path.join(visualization_dir, "Testing")
    plot_residual_kde(residuals_test_list, output_dir=testing_dir)
    scatter_residuals_vs_true(residuals_test_list, y_test, output_dir=testing_dir)
    #pca_residuals_visualization(test_datalog_primary, feature_columns, output_dir=testing_dir)
    #parallel_coordinates_residuals(test_datalog_primary, feature_columns, output_dir=testing_dir)
    # --------------------------------------- #
    # Validation Graphs #
    validation_dir = os.path.join(visualization_dir, "Validation")
    plot_residual_kde(residuals_validation_list, output_dir=validation_dir)
    scatter_residuals_vs_true(residuals_validation_list, y_validate, output_dir=validation_dir)
    #pca_residuals_visualization(validation_datalog_primary, feature_columns, output_dir=validation_dir)
    #parallel_coordinates_residuals(validation_datalog_primary, feature_columns, output_dir=validation_dir)
    # --------------------------------------- #
    print("\n *********************************************** \n FINISHED VISUALIZATIONS \n ***********************************************")
    
    end_time = datetime.now()
    print(f"Total Duration: {(end_time - start_time).total_seconds()} seconds")

def plot_and_save_results(x_train, y_train, y_train_pred, x_test, y_test, y_test_pred, func_name, mse_train, mse_test, directory, residuals_train, residuals_test):
    """
    Plots and saves residuals for training and testing data.

    Args:
        x_train, y_train: Training data and predictions
        x_test, y_test: Testing data and predictions
        func_name: Name of the function/dataset
        mse_train, mse_test: Mean Squared Error for training and testing data
        directory: Directory to save the plots
        residuals_train, residuals_test: Residuals for training and testing data

    Returns:
        None
    """

    # Calculate normalized residuals
    residuals_train = np.abs((y_train.reshape(-1) - y_train_pred.reshape(-1)) / y_train.reshape(-1))
    residuals_test = np.abs((y_test.reshape(-1) - y_test_pred.reshape(-1)) / y_test.reshape(-1))

    # Plot residuals for training and testing data
    plt.figure()
    plt.tight_layout()
    plt.scatter(np.abs(y_train), np.abs(residuals_train), alpha=0.75, color='b', label="Train Residuals")
    plt.scatter(np.abs(y_test), np.abs(residuals_test), alpha=0.75, color='r', label="Test Residuals")
    plt.xlabel("Normalized Abs True Values")
    plt.ylabel("Normalized Abs Residuals")
    plt.title(f"Residuals for {func_name} (Norm. Abs. Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(directory, f"{func_name}_residuals_abs.png"))
    plt.close()

def process_all_datasets(base_directory):
    """
    Processes all datasets in the specified base directory.

    Args:
    base_directory: The directory containing the datasets.
    """
    csv_files = [f for f in os.listdir(base_directory) if f.endswith('.csv')]
    validation_files = [f for f in os.listdir(base_directory + '/validation/') if f.endswith('.csv')]
    validation_set = pd.read_csv('')
    for file in csv_files:
        if file.endswith(".csv") and not file.endswith("_oos.csv"):
            df = pd.read_csv(os.path.join(base_directory, file))
            x = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values.reshape(-1, 1)
            func_name = file.split('.')[0]

            # Create a sub-directory for each dataset to save its results
            dataset_dir = os.path.join(base_directory, func_name)
            os.makedirs(dataset_dir, exist_ok=True)

            # Run the model on the dataset
            runModel(x, y, func_name, dataset_dir,validation_set)

def process_large_dataset(base_directory, dataset, sample_size, runtime, scale_boolean, troubleshooting_boolean):
    """
    Loads a large dataset, splits into train/test/validation, and runs model pipeline.
    Args:
        base_directory: Directory for dataset
        dataset: Dataset filename
        sample_size: Number of rows for train/test/validation
        runtime: Time limit for PySRRegressor
        scale_boolean: Whether to scale data
        troubleshooting_boolean: Whether to print troubleshooting info
    Returns:
        None
    """

    print("Running proccess_large_dataset...")

    # Create dataframe from dataset
    df = pd.read_csv(os.path.join(base_directory, dataset))

    # check if sample size exceeds dataset size
    if sample_size > len(df):
        raise ValueError(f"Sample size {sample_size} is larger than the number of rows in the DataFrame ({len(df)})")


    # randomly take X rows for test/train + validation
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)

    X_train_all, X_val, y_train_all, y_val = train_test_split(x, y, test_size=dataset_size, train_size=dataset_size, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train_all, y_train_all, test_size=int(dataset_size*0.8), train_size=int(dataset_size*0.2), random_state=42)

    func_name = dataset.split('.')[0] # make function name (without csv)
    feature_columns = df.columns[:-1]  # All columns except the last (features)
    target_column = df.columns[-1] # Last column (target)

    # Create a sub-directory for the dataset to save its results
    dataset_dir = os.path.join(base_directory, func_name)
    os.makedirs(dataset_dir, exist_ok=True)


    runModel(X_train, X_test, y_train, y_test, X_val, y_val, func_name, dataset_dir,feature_columns,target_column,runtime, scale_boolean, troubleshooting_boolean)

def process_large_dataset_with_validation(training_directory, training_dataset, validation_directory, validation_dataset, runtime, scale_boolean, troubleshooting_boolean):
    """
    Loads training and validation datasets, splits, and runs model pipeline.
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

    print("Running proccess_large_dataset_with_validation...")


    # Create dataframe from datasets
    training_df = pd.read_csv(os.path.join(training_directory, training_dataset))
    validation_df = pd.read_csv(os.path.join(validation_directory, validation_dataset))

    # check if sample size exceeds comfortable size dataset size
    if len(training_df) > 1000:
        print("Training dataset exceeds comfortable size, be prepared for slow runtime.")

    # Take X rows for test/train
    x = training_df.iloc[:, :-1].values
    y = training_df.iloc[:, -1].values.reshape(-1, 1)

    X_val = validation_df.iloc[:, :-1].values
    y_val = validation_df.iloc[:, -1].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=int(dataset_size*0.8), train_size=int(dataset_size*0.2), random_state=42)

    func_name = training_dataset.split('.')[0] # make function name (without csv)
    feature_columns = training_df.columns[:-1]  # All columns except the last (features)
    target_column = training_df.columns[-1] # Last column (target)

    # Create a sub-directory for the dataset to save its results
    dataset_dir = os.path.join(base_directory, func_name)
    os.makedirs(dataset_dir, exist_ok=True)


    runModel(X_train, X_test, y_train, y_test, X_val, y_val, func_name, dataset_dir,feature_columns,target_column,runtime, scale_boolean, troubleshooting_boolean)

def parallel_process_datasets(train_datasets, val_datasets, train_dir, val_dir, runtime, scale_boolean, troubleshooting_boolean):
    """
    Parallelizes processing of multiple train/validation dataset pairs.
    Args:
        train_datasets: list of training dataset filenames
        val_datasets: list of validation dataset filenames
        train_dir: directory containing training datasets
        val_dir: directory containing validation datasets
        runtime: time limit for each run
        scale_boolean: Whether to scale data
        troubleshooting_boolean: Whether to print troubleshooting info
    """
    with cf.ProcessPoolExecutor() as executor:
        futures = []
        for train_dataset, val_dataset in zip(train_datasets, val_datasets):
            print(f"Submitting: train={train_dataset}, val={val_dataset}")
            futures.append(
                executor.submit(
                    process_large_dataset_with_validation,
                    train_dir, train_dataset, val_dir, val_dataset, runtime, scale_boolean, troubleshooting_boolean
                )
            )
        for future in cf.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"A dataset run failed: {exc}")

# --- Feature Engineering Utilities --- #
# This function adds polynomial features (e.g., x, x^2, x1*x2, etc.) to your feature matrix X.
# It can help capture nonlinear relationships for models that benefit from such features.
def add_polynomial_features(X, degree=2):
    """
    Adds polynomial features up to the specified degree for each feature in X.
    Args:
        X: 2D numpy array of features
        degree: maximum degree of polynomial features to add
    Returns:
        X_poly: 2D numpy array with original and polynomial features
    """
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    return poly.fit_transform(X)

# --- Hyperparameter Tuning Utilities --- #
# This function runs a grid search over combinations of PySRRegressor hyperparameters.
# It launches each combination in parallel, so you can efficiently find the best settings.
# You must adapt runModel to accept and use the params dictionary for this to work fully.
def hyperparameter_grid_search(param_grid, train_datasets, val_datasets, train_dir, val_dir, runtime):
    """
    Runs a grid search over PySRRegressor hyperparameters in parallel.
    Args:
        param_grid: dict of parameter lists, e.g. {'population_size': [100, 500], 'maxsize': [20, 45]}
        train_datasets, val_datasets, train_dir, val_dir, runtime: as before
    Returns:
        results: list of (params, score) tuples
    """
    import itertools
    results = []
    keys, values = zip(*param_grid.items())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    def run_with_params(params, train_dataset, val_dataset):
        # You would need to pass params into PySRRegressor in runModel
        # This is a stub for illustration
        print(f"Running {train_dataset} with params {params}")
        # runModel(..., **params)
        return params, 0  # Replace 0 with actual score
    with cf.ProcessPoolExecutor() as executor:
        futures = []
        for params in param_combos:
            for train_dataset, val_dataset in zip(train_datasets, val_datasets):
                futures.append(executor.submit(run_with_params, params, train_dataset, val_dataset))
        for future in cf.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"A grid search run failed: {exc}")
    return results

# --- K-Fold Cross-Validation Utility --- #
# This function splits your data into k folds, trains and evaluates your model k times,
# each time using a different fold as the test set and the rest as training data.
# It returns a list of scores (e.g., NMSE or RMSE) for each fold.
def k_fold_cross_validation(X, y, k=5, model_func=None, **model_kwargs):
    """
    Perform k-fold cross-validation for a given model function.
    Args:
        X: Features (numpy array)
        y: Target (numpy array)
        k: Number of folds
        model_func: Function to train and evaluate model, must return score (e.g., NMSE)
        model_kwargs: Additional arguments for model_func
    Returns:
        scores: List of scores for each fold
    """
    from sklearn.model_selection import KFold
    scores = []
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # model_func should fit and evaluate the model, returning a score (e.g., NMSE)
        score = model_func(X_train, X_test, y_train, y_test, **model_kwargs)
        scores.append(score)
    return scores

def symbolic_regression_score(X_train, X_test, y_train, y_test, feature_columns, target_column, runtime, scale_boolean, troubleshooting_boolean):
    """
    Wrapper for k-fold cross-validation: trains symbolic regression and returns NMSE on test set.
    """
    from pysr import PySRRegressor
    from calculate_errors import calculate_mse_nmse
    # Optionally scale data here if needed
    if scale_boolean:
        from stat_plots import scale_data_Nuttii
        X_train, y_train, _, _ = scale_data_Nuttii(X_train, y_train)
        X_test, y_test, _, _ = scale_data_Nuttii(X_test, y_test)
    params = PYSR_PARAMS.copy()
    params["timeout_in_seconds"] = runtime
    model = PySRRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    _, nmse = calculate_mse_nmse(y_test, y_pred)
    return nmse

# --- Linear Regression Comparison --- #
# This function fits a linear regression model on the same data splits as your symbolic regression.
# It saves metrics, predictions, and residual plots for direct comparison.
def run_linear_regression_and_compare(x_train, x_test, y_train, y_test, x_validate, y_validate, feature_columns, target_column, visualization_dir):
    """
    Fit a linear regression on the same data and save results for comparison.
    Args:
        x_train, x_test, y_train, y_test, x_validate, y_validate: Data splits
        feature_columns: list of feature names
        target_column: name of target
        output_dir: directory to save results
    """
    from sklearn.linear_model import LinearRegression
    from calculate_errors import calculate_mse_nmse, calculate_rmse

    os.makedirs(visualization_dir, exist_ok=True)
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    y_validate_pred = model.predict(x_validate)

    mse_train, nmse_train = calculate_mse_nmse(y_train, y_train_pred)
    mse_test, nmse_test = calculate_mse_nmse(y_test, y_test_pred)
    mse_validate, nmse_validate = calculate_mse_nmse(y_validate, y_validate_pred)
    rmse_train = calculate_rmse(y_train, y_train_pred)
    rmse_test = calculate_rmse(y_test, y_test_pred)
    rmse_validate = calculate_rmse(y_validate, y_validate_pred)

    # Save results
    parent_dir = os.path.dirname(visualization_dir)
    with open(os.path.join(parent_dir, "linear_regression_results.txt"), "w") as f:
        f.write(f"Linear Regression Results\n")
        f.write(f"Train MSE: {mse_train}\n")
        f.write(f"Train NMSE: {nmse_train}\n")
        f.write(f"Train RMSE: {rmse_train}\n")
        f.write(f"Test MSE: {mse_test}\n")
        f.write(f"Test NMSE: {nmse_test}\n")
        f.write(f"Test RMSE: {rmse_test}\n")
        f.write(f"Validation MSE: {mse_validate}\n")
        f.write(f"Validation NMSE: {nmse_validate}\n")
        f.write(f"Validation RMSE: {rmse_validate}\n")
        f.write(f"Coefficients: {model.coef_}\n")
        f.write(f"Intercept: {model.intercept_}\n")

    # Optionally, save predictions for further analysis
    pd.DataFrame(y_train_pred, columns=[f"{target_column}_pred"]).to_csv(os.path.join(parent_dir, "linear_train_pred.csv"), index=False)
    pd.DataFrame(y_test_pred, columns=[f"{target_column}_pred"]).to_csv(os.path.join(parent_dir, "linear_test_pred.csv"), index=False)
    pd.DataFrame(y_validate_pred, columns=[f"{target_column}_pred"]).to_csv(os.path.join(parent_dir, "linear_validate_pred.csv"), index=False)

    # Optionally, plot residuals
    import matplotlib.pyplot as plt
    residuals_train = y_train.flatten() - y_train_pred.flatten()
    residuals_test = y_test.flatten() - y_test_pred.flatten()
    plt.figure()
    plt.scatter(y_train, residuals_train, alpha=0.5, label='Train')
    plt.scatter(y_test, residuals_test, alpha=0.5, label='Test')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('True Value')
    plt.ylabel('Residual')
    plt.title('Linear Regression Residuals')
    plt.legend()
    plt.savefig(os.path.join(visualization_dir, "linear_regression_residuals.png"))
    plt.close()
# --- Hyperparameter Tuning Utilities --- #
def hyperparameter_grid_search(param_grid, train_datasets, val_datasets, train_dir, val_dir, runtime):
    """
    Runs a grid search over PySRRegressor hyperparameters in parallel.
    Args:
        param_grid: dict of parameter lists, e.g. {'population_size': [100, 500], 'maxsize': [20, 45]}
        train_datasets, val_datasets, train_dir, val_dir, runtime: as before
    Returns:
        results: list of (params, score) tuples
    """
    import itertools
    results = []
    keys, values = zip(*param_grid.items())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    def run_with_params(params, train_dataset, val_dataset):
        # You would need to pass params into PySRRegressor in runModel
        # This is a stub for illustration
        print(f"Running {train_dataset} with params {params}")
        # runModel(..., **params)
        return params, 0  # Replace 0 with actual score
    with cf.ProcessPoolExecutor() as executor:
        futures = []
        for params in param_combos:
            for train_dataset, val_dataset in zip(train_datasets, val_datasets):
                futures.append(executor.submit(run_with_params, params, train_dataset, val_dataset))
        for future in cf.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"A grid search run failed: {exc}")
    return results

def k_fold_cross_validation(X, y, k=5, model_func=None, **model_kwargs):
    """
    Perform k-fold cross-validation for a given model function.
    Args:
        X: Features (numpy array)
        y: Target (numpy array)
        k: Number of folds
        model_func: Function to train and evaluate model, must return score (e.g., NMSE)
        model_kwargs: Additional arguments for model_func
    Returns:
        scores: List of scores for each fold
    """
    from sklearn.model_selection import KFold
    scores = []
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        score = model_func(X_train, X_test, y_train, y_test, **model_kwargs)
        scores.append(score)
    return scores

if __name__ == "__main__":
    if batch_boolean:
        noise_folders = [f for f in os.listdir(base_directory) if f.endswith('Noise')]
        for folder in noise_folders:
            location = base_directory + '/' + folder
            csv_files = [f for f in os.listdir(location) if f.endswith('.csv')]
            for file in csv_files:
                print(f'folder: {folder}')
                print(f'file: {file}')
                process_large_dataset(location, file, dataset_size, runtime, scale_boolean, troubleshooting_boolean)
    else:
        os.chdir(base_directory)
        if separate_validation and not val_batch_boolean:
            process_large_dataset_with_validation(train_directory, training_datasets[0], val_directory, validation_datasets[0], runtime, scale_boolean, troubleshooting_boolean)
        elif val_batch_boolean:
            parallel_process_datasets(training_datasets, validation_datasets, train_directory, val_directory, runtime, scale_boolean, troubleshooting_boolean)
        else:
            process_large_dataset(base_directory, training_datasets[0], dataset_size, runtime, scale_boolean, troubleshooting_boolean)
