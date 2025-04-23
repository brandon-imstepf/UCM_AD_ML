import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from pysr import PySRRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.metrics import r2_score
import tkinter as tk
from tkinter import messagebox
from stat_plots import *
from calculate_errors import *

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


''' // For v5 // 
Need to:
    - Improve data visualization
        - Add true vs predicted plots ( x = y )
    - Record mean, SD for z-score transformation
    - add RMSE +
    - add R^2 +
        -  sklearn.metrics.r2_score(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average', force_finite=True)
    - Colors:
        lin_color = 'orange'
        joint_color = '#1F77B4'
'''

base_directory = 'C:/Users/brand/Desktop/Raj-Sindi/training_data/sim_csv_v9_fixed'
dataset_name = 'data_frac_07_widerange.csv'

def runModel(x, y, func_name, directory,validation_dataset):
    # Change the working directory to the target directory
    os.chdir(directory)

    scale_boolean = True
    troubleshooting_boolean = False


    if scale_boolean:
        # scale data to make it easier
        #x, y = scale_data_ZScore(x, y)
        x,y,PredScaleFit,TargetScaleFit = scale_data_Nuttii(x,y)


    # Split data into training and testing sets (80/20 split)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_validate = validation_dataset.iloc[:, :-1].values
    y_validate = validation_dataset.iloc[:, -1].values.reshape(-1, 1)
    feature_columns = validation_dataset.columns[:-1]  # All columns except the last (features)
    target_column = validation_dataset.columns[-1]     # The last column (target)

    if scale_boolean:
        # Ensure scaling consistency
        #x_validate, y_validate = scale_data_ZScore(x_validate, y_validate)
        x_validate, y_validate, PredScaleFit_validate,TargetScaleFit_validate = scale_data_Nuttii(x_validate,y_validate)
    
    # Learn equations using PySR
    model = PySRRegressor(
        procs=4,  # Number of processes
        populations=32,  # Number of populations in the evolutionary algorithm
        population_size=500,  # Size of each population
        ncycles_per_iteration=10000,  # Number of cycles per iteration
        niterations=100000,  # Maximum number of iterations
        complexity_of_constants=5,
        #parsimony = 0.01,
        #adaptive_parsimony_scaling = 1000,
         constraints={
         "/": (-1, 9),
         "^": (-1, 1),
         "square": 9,
         "cube": 9,
         "exp": 9,
         },
        early_stop_condition=("stop_if(loss,complexity) = loss < 1e-5 && complexity < 20 || loss < 1e-9"),  # Early stopping condition
        timeout_in_seconds=60*5,  # Timeout in seconds/minutes/hours
        maxsize=50,  # Maximum size of the equations
        maxdepth=5,  # Maximum depth of the equations
        binary_operators=["*", "+", "-", "/","^"],  # Binary operators to be used
        unary_operators=["sin","cos","exp","log"],  # Unary operators to be used
        #nested_constraints={
        #    "square": {"square": 1, "cube": 0, "exp": 0},
        #    "cube": {"square": 1, "cube": 0, "exp": 0},
        #    "exp": {"square": 1, "cube": 1, "exp": 0},
        #},  # Constraints on nesting unary operators
        #select_k_features=4,  # Number of features to be selected
        progress=False,  # Show progress
        #weight_randomize=5,  # Randomization weight
        weight_add_node=2,
        weight_optimize=0.001,
        precision=32,  # Precision of the calculations
        warm_start=False,  # Use the previous model as a warm start
        turbo=False,  # Use turbo mode for faster computations
        denoise=False,  # Denoiser (Gaussian white noise kernel)
        model_selection="best",  # Save the best equations
    )

    start_time = datetime.now()

    model.fit(x_train, y_train)

    # Select the best equation based on score and loss
    global best_idx
    best_idx = model.equations_.query(f"loss < {2 * model.equations_.loss.min()}").score.idxmax()
    #equation = model.equations_.iloc[[best_idx]]

    # Prepare directory for saving results
    equation_dir = os.path.join(directory, "Info")
    os.makedirs(equation_dir, exist_ok=True)
    
    # for BEST equation #
    best_eq = model.sympy(best_idx)
    best_y_train_pred = model.predict(x_train, index=best_idx)
    best_y_test_pred = model.predict(x_test, index=best_idx)

    best_mse_train, best_nmse_train = calculate_mse_nmse(y_train, best_y_train_pred)
    best_mse_test, best_nmse_test = calculate_mse_nmse(y_test,best_y_test_pred)

    best_residuals_train = np.abs(y_train.flatten() - best_y_train_pred)
    best_residuals_test = np.abs(y_test.flatten() - best_y_test_pred)
    # ----------------------------------------------------------- #

    # Write equation and timing to log
    with open(os.path.join(equation_dir, "best.txt"), "a") as file:
        file.write(f"Selected Best Function: {best_idx}, {best_eq}\n")
        file.write(f"Train MSE: {best_mse_train}\n")
        file.write(f"Train NMSE: {best_nmse_train}\n")
        file.write(f"Test MSE: {best_mse_test}\n")
        file.write(f"Test NMSE: {best_nmse_test}\n")
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
    r2_train_list, r2_test_list, r2_validate_list = [],[],[]       

    # Inverse Transform the targets
    if scale_boolean:
        y_train = TargetScaleFit.inverse_transform(y_train)
        y_test = TargetScaleFit.inverse_transform(y_test)
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
            y_train_pred = TargetScaleFit.inverse_transform(y_train_pred).flatten()  # should be (800,1)
            print(f'The shape of y_train_pred before reshaping the transformation is: {np.shape(y_train_pred)}')
            
            y_test_pred = y_test_pred.reshape(-1, 1)  # should be (200,)
            #print(f'The shape of y_test_pred AFTER reshaping is: {np.shape(y_test_pred)}')
            y_test_pred = TargetScaleFit.inverse_transform(y_test_pred).flatten()  # should be (200,)

            y_validation_pred = y_validation_pred.reshape(-1, 1)  # should be (1000,)
            #print(f'The shape of y_validation_pred AFTER reshaping is: {np.shape(y_validation_pred)}')
            y_validation_pred = TargetScaleFit_validate.inverse_transform(y_validation_pred).flatten()  # should be (1000,)

        mse_train, nmse_train = calculate_mse_nmse(y_train, y_train_pred)
        mse_test, nmse_test = calculate_mse_nmse(y_test, y_test_pred)
        mse_validation, nmse_validation = calculate_mse_nmse(y_validate, y_validation_pred)

        residuals_train = np.abs(y_train.flatten() - y_train_pred)
        residuals_test = np.abs(y_test.flatten() - y_test_pred)
        residuals_validation = np.abs(y_validate.flatten() - y_validation_pred)  

        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        r2_validate = r2_score(y_validate, y_validation_pred)      


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
        r2_train_list.append(r2_train)
        r2_test_list.append(r2_test)
        r2_validate_list.append(r2_validate)

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
        'R2_Train' : r2_train_list,
    })

    test_datalog_secondary = pd.DataFrame({
        'Model': [f'Model_{i+1}' for i in range(len(mse_test_list))],
        'MSE_Test': mse_test_list,
        'NMSE_Test': nmse_test_list,
        'R2_Test': r2_test_list,
    })

    validation_datalog_secondary = pd.DataFrame({
        'Model': [f'Model_{i+1}' for i in range(len(mse_validation_list))],
        'MSE_Validation': mse_validation_list,
        'NMSE_Validation': nmse_validation_list,
        'R2_Validation': r2_validate_list,
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
    # --------------------------------------- #
    # Training Graphs #
    training_dir = os.path.join(visualization_dir, "Training")
    plot_residual_kde(residuals_train_list, output_dir=training_dir)
    scatter_residuals_vs_true(residuals_train_list, y_train, output_dir=training_dir)
    scatter_best_residuals_vs_true(residuals_train_list, y_train, best=best_idx, output_dir=training_dir)
    #pca_residuals_visualization(train_datalog_primary, feature_columns, output_dir=training_dir)
    #parallel_coordinates_residuals(train_datalog_primary, feature_columns, output_dir=training_dir)
    # --------------------------------------- #
    # Testing Graphs #
    testing_dir = os.path.join(visualization_dir, "Testing")
    plot_residual_kde(residuals_test_list, output_dir=testing_dir)
    scatter_residuals_vs_true(residuals_test_list, y_test, output_dir=testing_dir)
    scatter_best_residuals_vs_true(residuals_test_list, y_test, best=best_idx, output_dir=testing_dir)
    #pca_residuals_visualization(test_datalog_primary, feature_columns, output_dir=testing_dir)
    #parallel_coordinates_residuals(test_datalog_primary, feature_columns, output_dir=testing_dir)
    # --------------------------------------- #
    # Validation Graphs #
    validation_dir = os.path.join(visualization_dir, "Validation")
    plot_residual_kde(residuals_validation_list, output_dir=validation_dir)
    scatter_residuals_vs_true(residuals_validation_list, y_validate, output_dir=validation_dir)
    scatter_best_residuals_vs_true(residuals_validation_list, y_validate, best=best_idx, output_dir=validation_dir)
    #pca_residuals_visualization(validation_datalog_primary, feature_columns, output_dir=validation_dir)
    #parallel_coordinates_residuals(validation_datalog_primary, feature_columns, output_dir=validation_dir)
    # --------------------------------------- #
    print("\n *********************************************** \n FINISHED VISUALIZATIONS \n ***********************************************")
    


    # troubleshoot first #
    #print(datalog)
    #print(f"The shape of the datalog is: {datalog.shape}")
    ######################
    # Compare values with Linear regression and Polynomial regression
    #lr_x_train, lr_x_test, lr_y_train, lr_y_test, lr_x_mse, lr_y_mse = Do_LinearRegression(x,y)
    #pr_x_train, pr_x_test, pr_y_train, pr_y_test, pr_x_mse, pr_y_mse = Do_PolynomialRegression(x,y)


    # ----------------------------------------------------------- #
    # ----------------------------------------------------------- #



    # Plotting and saving results !!!!!!!!!!!DEPRECATED!!!!!!!!!!!!
    #plot_and_save_results(x_train, y_train, y_train_pred, x_test, y_test, y_test_pred, func_name, mse_train, mse_test, equation_dir, residuals_train, residuals_test)
    #plot_complexity_vs_error(complexity, nmse_list, oos_nmse_list, save_path)

    end_time = datetime.now()
    print(f"Total Duration: {(end_time - start_time).total_seconds()} seconds")

def plot_and_save_results(x_train, y_train, y_train_pred, x_test, y_test, y_test_pred, func_name, mse_train, mse_test, directory, residuals_train, residuals_test):
    """
    Plots and saves the results including residuals on both training and testing data.

    Args:
    - x_train, y_train: Training data and predictions.
    - x_test, y_test: Testing data and predictions.
    - func_name: Name of the function/dataset.
    - mse_train, mse_test: Mean Squared Error for training and testing data.
    - directory: Directory to save the plots.
    - residuals_train, residuals_test: Residuals for training and testing data.
    
    Returns:
    - None
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

def process_large_dataset(base_directory, dataset, sample_size):
    '''
    Input:
        base_directory: the directory the dataset is contained in
        dataset: the filename of the dataset
        size: how many rows of the csv to use for test/train = validation 
                e.g. size = 1000 means 1000 for test/train split and 1000 for validation, sampled without replacement
    Output:
        None -- the RunModel function itself creates all graphs and data.
    '''
    print("Running proccess_large_dataset...")

    # Create dataframe from dataset
    df = pd.read_csv(os.path.join(base_directory, dataset))

    # check if sample size exceeds dataset size
    if sample_size > len(df):
        raise ValueError(f"Sample size {sample_size} is larger than the number of rows in the DataFrame ({len(df)})")


    # randomly take X rows for test/train + validation
    np.random.seed(42) # for reproducibility
    random_indices = np.random.choice(df.index, size=sample_size*2, replace=False)
    train_test_indices = random_indices[:sample_size]
    validation_indices = random_indices[sample_size:]
    train_test_set = df.loc[train_test_indices]
    validation_set = df.loc[validation_indices]



    x = train_test_set.iloc[:, :-1].values
    y = train_test_set.iloc[:, -1].values.reshape(-1, 1)



    feature_columns = df.columns[:-1]  # All columns except the last (features)
    target_column = df.columns[-1] # Last column (target)
    func_name = dataset.split('.')[0] # make function name (without csv)

    # Create a sub-directory for the dataset to save its results
    dataset_dir = os.path.join(base_directory, func_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # troubleshooting
    print(f"Train/Test Set Shape: {train_test_set.shape}")
    print(f"Validation Set Shape: {validation_set.shape}")

    runModel(x, y, func_name, dataset_dir, validation_set)


# Batch
batch_boolean = False

if batch_boolean:
    #base_directory = "C:/Users/brand/Documents/Python Scripts/pysr_code/datasets/Nguyen-RBFs-ns-oos-(1,3,20)"dso_paper_datasets
    base_directory = "C:/Users/brand/Documents/Python Scripts/pysr_code/datasets/Nguyen-RBFs-n-ns-(1,100,1000)"
    #process_all_datasets(base_directory)

    noise_folders = [f for f in os.listdir(base_directory) if f.endswith('Noise')]
    for folder in noise_folders:
        location = base_directory + '/' + folder
        csv_files = [f for f in os.listdir(location) if f.endswith('.csv')]
        for file in csv_files:
            print(f'folder: {folder}')
            print(f'file: {file}')
            process_large_dataset(location,file,100)

# Single
process_large_dataset(base_directory,dataset_name,1000)

os.chdir(base_directory)


#process_all_datasets(base_directory)
#os.chdir(base_directory)

'''
csv_files = [f for f in os.listdir(base_directory) if f.endswith('.csv')]

for file in csv_files:
    if file.endswith(".csv") and not file.endswith("_oos.csv"):
        process_large_dataset(base_directory,file,1000)

os.chdir(base_directory)
'''


'''
##################################################
Old elbow curve graph. Now done within RunModel().
##################################################
# Now, go into the subfolder and find the csv(s).
dataset_path = find_csv(base_directory)
print(f"dataset_path = {dataset_path}")
# create elbow plot(s)
complexity, loss_list, function_list, oos_loss_list = read_hof_csv(dataset_path, oos_csv=None)
plot_complexity_vs_loss(complexity, loss_list, oos_loss_list=None, save_path=base_directory + "/",best=best_idx)
plot_complexity_vs_loss_loglog(complexity, loss_list, oos_loss_list=None, save_path=base_directory + "/",best=best_idx)
# ----------------------------------------------------------- #
'''

'''
#temporary subdir searching bandaid
subdir = [3,5,10,25,50]
for noise in subdir:
    print(noise)
    base_directory = "C:/Users/brand/Documents/Python Scripts/pysr_code/datasets/Imstepf-RBFs-ns-oos-(3,5,20)/" + str(noise) + "_Percent_Noise"
    process_all_datasets(base_directory)
'''
