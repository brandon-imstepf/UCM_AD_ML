import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import sympy as sp
from sklearn import metrics,preprocessing, svm 
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA 
from calculate_errors import calculate_mse
import seaborn as sns
from matplotlib.colors import Normalize

# ------------------------------------ #
# 1. Residual Distribution Analysis
def plot_residual_kde(residuals_list, output_dir):
    """
    Plots KDEs of residuals for each model.

    Parameters:
        datalog (pd.DataFrame): Primary datalog containing residual columns for each model.
        output_dir (str): Directory to save the generated plot.

    Outputs:
        Saves a KDE plot of residuals for each model in the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    # Create a subplot
    ax = plt.subplot(111)
    for idx, residuals in enumerate(residuals_list):
        sns.kdeplot(residuals, label=f"Model {idx + 1}", alpha=0.6)
    # Adjust plot area to fit the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])  # Reduce plot width
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Place legend to the right
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("Residual Distribution (KDE)")
    ax.set_xlabel("Log Residuals")
    ax.set_ylabel("Log Density")
    # Add the legend outside the plot
    plt.savefig(os.path.join(output_dir, "residual_kde.png"))
    plt.close()

# ------------------------------------ #
# 2. Residuals vs. True Values
def scatter_residuals_vs_true(residual_list, trueval_list, output_dir):
    """
    Creates scatter plots of residuals vs. true values for each model.

    Parameters:
        datalog (pd.DataFrame): Primary datalog containing `y_true (flux)` and residual columns.
        output_dir (str): Directory to save the generated plot.

    Outputs:
        Saves scatter plots for residuals vs. `y_true (flux)` in the output directory.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    # Create a subplot
    ax = plt.subplot(111)

    # Plot each residual column
    for residual_col in len(residual_list):
        ax.scatter(residual_list, trueval_list, label=residual_col, alpha=0.6)

    # Add a horizontal line at y=0
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.8)

    # Adjust plot area to fit the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])  # Reduce plot width
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Place legend to the right

    # Set title and axis labels
    ax.set_title("Residuals vs. True Values")  
    ax.set_xlabel("True Values (Flux)")       
    ax.set_ylabel("Residuals")                

    # Save the plot
    plt.savefig(os.path.join(output_dir, "residuals_vs_true_values.png"), bbox_inches="tight")
    plt.close()

# ------------------------------------ #
# 3. Feature-Target Heatmap
def heatmap_feature_target(datalog, feature_x, feature_y, target_col, output_dir):
    """
    Creates a heatmap of the relationship between two features and the target.

    Parameters:
        datalog (pd.DataFrame): Primary datalog containing feature and target columns.
        feature_x (str): Feature name for x-axis.
        feature_y (str): Feature name for y-axis.
        target_col (str): Target column name.
        output_dir (str): Directory to save the generated heatmap.

    Outputs:
        Saves a heatmap plot showing the relationship between two features and the target.
    """
    plt.figure(figsize=(8, 6))
    # Create a subplot
    ax = plt.subplot(111)
    heatmap_data = datalog.pivot_table(index=feature_y, columns=feature_x, values=target_col, aggfunc='mean')
    sns.heatmap(heatmap_data, cmap='coolwarm')
     # Adjust plot area to fit the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])  # Reduce plot width
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Place legend to the right
    plt.title(f"Heatmap of {target_col} vs. {feature_x} and {feature_y}")
    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    plt.savefig(os.path.join(output_dir, f"heatmap_{feature_x}_{feature_y}.png"))
    plt.close()

# ------------------------------------ #
# 4. Predictions vs. True Values
def scatter_predictions_vs_true(datalog, output_dir):
    """
    Creates scatter plots of predictions vs. true values for each model.

    Parameters:
        datalog (pd.DataFrame): Primary datalog containing `y_true (flux)` and prediction columns.
        output_dir (str): Directory to save the generated plot.

    Outputs:
        Saves scatter plots for predictions vs. `y_true (flux)` in the output directory.
    """
    prediction_columns = [col for col in datalog.columns if '_Y_pred' in col]
    plt.figure(figsize=(10, 6))
    # Create a subplot
    ax = plt.subplot(111)
    for col in prediction_columns:
        plt.scatter(datalog['y_true (flux)'], datalog[col], alpha=0.5, label=col)
    plt.plot(datalog['y_true (flux)'], datalog['y_true (flux)'], color='black', linestyle='--', label='y = x')
    # Adjust plot area to fit the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])  # Reduce plot width
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Place legend to the right
    ax.set_title("Predictions vs. True Values")
    ax.set_xlabel("True Values (Flux)")
    ax.set_ylabel("Predicted Values")
    plt.savefig(os.path.join(output_dir, "predictions_vs_true.png"))
    plt.close()

# ------------------------------------ #
# 5. Residual Heatmap
def heatmap_residuals(datalog, feature_x, feature_y, residual_col, output_dir):
    """
    Creates a heatmap showing the relationship between two features and residuals.

    Parameters:
        datalog (pd.DataFrame): Primary datalog containing feature and residual columns.
        feature_x (str): Feature name for x-axis.
        feature_y (str): Feature name for y-axis.
        residual_col (str): Residual column name.
        output_dir (str): Directory to save the generated heatmap.

    Outputs:
        Saves a heatmap plot showing residuals averaged over the two features.
    """
    plt.figure(figsize=(8, 6))
    heatmap_data = datalog.pivot_table(index=feature_y, columns=feature_x, values=residual_col, aggfunc='mean')
    sns.heatmap(heatmap_data, cmap='coolwarm', norm=Normalize(-1, 1))
    plt.title(f"Heatmap of Residuals ({residual_col}) vs. {feature_x} and {feature_y}")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.savefig(os.path.join(output_dir, f"residual_heatmap_{feature_x}_{feature_y}.png"))
    plt.close()

# ------------------------------------ #
# 6. Pairwise Feature Relationships
def pairplot_residuals_colored(datalog, output_dir):
    """
    Creates pair plots of features colored by residuals.

    Parameters:
        datalog (pd.DataFrame): Primary datalog containing features and residuals.
        output_dir (str): Directory to save the generated pair plot.

    Outputs:
        Saves a pair plot where features are colored by residual magnitudes.
    """
    residual_columns = [col for col in datalog.columns if 'Residuals' in col]
    if residual_columns:
        datalog['Residual Magnitude'] = datalog[residual_columns].abs().mean(axis=1)
        sns.pairplot(datalog, vars=['gamma1', 'lambda1', 'delta', 'epsilon'], hue='Residual Magnitude', palette='coolwarm')
        plt.savefig(os.path.join(output_dir, "pairplot_residuals_colored.png"))
        plt.close()

# ------------------------------------ #
# 7. Time-Series Style Plots
def time_series_predictions_vs_true(datalog, output_dir):
    """
    Creates a time-series style plot of true values and predictions.

    Parameters:
        datalog (pd.DataFrame): Primary datalog containing `y_true (flux)` and prediction columns.
        output_dir (str): Directory to save the generated plot.

    Outputs:
        Saves a time-series plot of predictions and true values in the output directory.
    """
    prediction_columns = [col for col in datalog.columns if '_Y_pred' in col]
    plt.figure(figsize=(12, 6))
    plt.plot(datalog.index, datalog['y_true (flux)'], label='True Values', linewidth=2)
    for col in prediction_columns:
        plt.plot(datalog.index, datalog[col], label=col, alpha=0.7)
    plt.title("Time-Series Predictions vs. True Values")
    plt.xlabel("Index")
    plt.ylabel("Flux")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "time_series_predictions_vs_true.png"))
    plt.close()

# ------------------------------------ #
# 8. Binned Residual Analysis
def binned_residuals_analysis(datalog, feature_col, residual_col, bins, output_dir):
    """
    Creates a plot of average residuals binned by a specific feature.

    Parameters:
        datalog (pd.DataFrame): Primary datalog containing feature and residual columns.
        feature_col (str): Feature column name to bin residuals by.
        residual_col (str): Residual column name.
        bins (int): Number of bins to use for the feature.
        output_dir (str): Directory to save the generated plot.

    Outputs:
        Saves a plot showing binned residuals in the output directory.
    """
    binned = pd.cut(datalog[feature_col], bins=bins)
    residual_means = datalog.groupby(binned)[residual_col].mean()
    bin_centers = [interval.mid for interval in binned.cat.categories]
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, residual_means, width=0.1 * (max(bin_centers) - min(bin_centers)), color='skyblue', edgecolor='black')
    plt.title(f"Binned Residuals Analysis: {feature_col} vs. {residual_col}")
    plt.xlabel(feature_col)
    plt.ylabel("Average Residual")
    plt.savefig(os.path.join(output_dir, f"binned_residuals_{feature_col}.png"))
    plt.close()

# ------------------------------------ #
# 9. Parallel Coordinates Plot
def parallel_coordinates_residuals(datalog, output_dir):
    """
    Creates a parallel coordinates plot showing feature relationships and residuals.

    Parameters:
        datalog (pd.DataFrame): Primary datalog containing features and residuals.
        output_dir (str): Directory to save the generated plot.

    Outputs:
        Saves a parallel coordinates plot in the output directory.
    """
    from pandas.plotting import parallel_coordinates

    features = ['gamma1', 'lambda1', 'delta', 'epsilon', 'NRow', 'NCol']
    residual_columns = [col for col in datalog.columns if 'Residuals' in col]
    for residual_col in residual_columns:
        temp_data = datalog[features + [residual_col]].copy()
        temp_data['Residual Magnitude'] = np.abs(temp_data[residual_col])
        temp_data['Residual Category'] = pd.cut(temp_data['Residual Magnitude'], bins=3, labels=['Low', 'Medium', 'High'])
        plt.figure(figsize=(12, 6))
        parallel_coordinates(temp_data, class_column='Residual Category', cols=features, colormap='coolwarm')
        plt.title(f"Parallel Coordinates: Features Colored by {residual_col}")
        plt.savefig(os.path.join(output_dir, f"parallel_coordinates_{residual_col}.png"))
        plt.close()

# ------------------------------------ #
# 10. Dimensionality Reduction (PCA)
def pca_residuals_visualization(datalog, features, output_dir):
    """
    Reduces feature dimensions using PCA and visualizes residual magnitudes in 2D.

    Parameters:
        datalog (pd.DataFrame): Primary datalog containing features and residuals.
        output_dir (str): Directory to save the generated PCA plot.

    Outputs:
        Saves a 2D PCA scatter plot in the output directory.
    """
    residual_columns = [col for col in datalog.columns if 'Residuals' in col]
    if residual_columns:
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(datalog[features])
        datalog['PCA1'], datalog['PCA2'] = reduced_data[:, 0], reduced_data[:, 1]
        residual_magnitude = datalog[residual_columns].abs().mean(axis=1)
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(datalog['PCA1'], datalog['PCA2'], c=residual_magnitude, cmap='coolwarm', alpha=0.8)
        plt.colorbar(scatter, label='Residual Magnitude')
        plt.title("PCA Visualization of Features with Residuals")
        plt.xlabel("PCA1")
        plt.ylabel("PCA2")
        plt.savefig(os.path.join(output_dir, "pca_residuals_visualization.png"))
        plt.close()

# ------------------------------------ #
# 11. Combined Visualizations
def combined_predictions_vs_true_and_residuals(datalog, output_dir):
    """
    Creates a combined scatter plot of predictions vs. true values and residuals.

    Parameters:
        datalog (pd.DataFrame): Primary datalog containing `y_true (flux)`, predictions, and residuals.
        output_dir (str): Directory to save the generated combined plot.

    Outputs:
        Saves a combined plot of predictions vs. true values and residuals.
    """
    prediction_columns = [col for col in datalog.columns if '_Y_pred' in col]
    residual_columns = [col for col in datalog.columns if 'Residuals' in col]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Predictions vs. True
    for col in prediction_columns:
        axes[0].scatter(datalog['y_true (flux)'], datalog[col], alpha=0.5, label=col)
    axes[0].plot(datalog['y_true (flux)'], datalog['y_true (flux)'], color='black', linestyle='--', label='y = x')
    axes[0].set_title("Predictions vs. True Values")
    axes[0].set_xlabel("True Values (Flux)")
    axes[0].set_ylabel("Predicted Values")
    axes[0].legend()

    # Residuals
    for col in residual_columns:
        axes[1].scatter(datalog['y_true (flux)'], datalog[col], alpha=0.5, label=col)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[1].set_title("Residuals vs. True Values")
    axes[1].set_xlabel("True Values (Flux)")
    axes[1].set_ylabel("Residuals")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_predictions_and_residuals.png"))
    plt.close()

def Do_LinearRegression(x,y):
    print("LinearRegression")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25) 
    
    regr = LinearRegression().fit(x_train, y_train)
    print(f"Linear Regression gives a score of: {regr.score(x_test, y_test)}")
    # get MSE to return
    x_mse = calculate_mse(x_train,x_test)
    y_mse = calculate_mse(y_train,y_test)
    return x_train, x_test, y_train, y_test, x_mse, y_mse
      
def Do_PolynomialRegression():
    print("PolynomialRegression")

def scale_data_YeoJohnson(x, y):
    # Apply Yeo-Johnson power transformation (works with both positive and negative data)
    power_transformer = PowerTransformer(method='yeo-johnson')
    x = power_transformer.fit_transform(x)
    y = power_transformer.fit_transform(y.reshape(-1, 1)).flatten()  # Flatten y after transforming

    # Scale the data to a standard range (e.g., zero mean, unit variance)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    x = scaler_x.fit_transform(x)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    return x, y

def scale_data_ZScore(x, y):
    # Apply ZScaling
    # Initialize the scaler
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    # Fit and transform the data
    standardized_x = scaler_x.fit_transform(x)
    standardized_y = scaler_y.fit_transform(y)

    return standardized_x, standardized_y

# Function to generate a single plot comparing true and predicted values
def plot_single_comparison(x, y_true, y_pred, equation_str, complexity, nmse, mse, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, label='True Function', color='blue', linewidth=2)
    plt.plot(x, y_pred, label='Predicted Function', linestyle='--', color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Comparison: True vs Predicted\nEquation: {equation_str}')
    plt.legend()
    # Add text box with complexity, NMSE, and MSE details
    plt.gcf().text(0.15, 0.7, f"Complexity: {complexity}\nNMSE: {nmse:.4f}\nMSE: {mse:.4f}",
                   fontsize=10, va='top', bbox=dict(boxstyle="round", edgecolor="black", facecolor="white"))
    plt.savefig(save_path)
    plt.close()

# Function to plot complexity vs error (NMSE)
def plot_complexity_vs_error(complexity, nmse_list, oos_nmse_list=None, save_path=None, best=None):
    '''
    Parameters:
    -----------
    complexity: L(E) = L_pred(E) + (parsimony) · C(E), for C(E) the complexity of an expression E and L_pred the predictive loss.
                Automatically output by PySR and accessible in the Hall of Fame file.
    nmse_list: list of normalized mean squared errors
    oos_loss_list: Optional, if you have an out-of-sample list of loss errors to compare against.
    save_path: Directory for saving the graph (optional).
    best: Optional integer indicating the "best" function index to highlight in the graph.
    '''
    plt.figure(figsize=(8, 6))

    # Plot in-sample losses
    plt.scatter(complexity, nmse_list, label='In-Sample NMSE', color='green')
    plt.plot(complexity, nmse_list, color='green', alpha=0.5)

    # Plot out-of-sample losses if available
    if oos_nmse_list:
        plt.scatter(complexity, oos_nmse_list, label='Out-of-Sample NMSE', color='red')
        plt.plot(complexity, oos_nmse_list, color='red', alpha=0.5)

    # Highlight the "best" function if provided
    if best is not None and 0 <= best < len(complexity):
        plt.scatter(complexity[best], nmse_list[best], color='blue', s=100, zorder=5, label='Best Function')
        plt.annotate(f"Best",
                     (complexity[best], nmse_list[best]),
                     textcoords="offset points",
                     xytext=(10, 10),
                     ha='center',
                     fontsize=10,
                     arrowprops=dict(facecolor='blue', arrowstyle="->"))

    # Graph labels and styling
    plt.xlabel('Complexity')
    plt.ylabel('NMSE')
    plt.title('Complexity vs Normalized Mean Square Error (NMSE)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_complexity_vs_error_loglog(complexity, nmse_list, oos_nmse_list=None, save_path=None,best=None):
    '''
    Parameters:
    -----------
    complexity: L(E) = L_pred(E) + (parsimony) · C(E), for C(E) the complexity of an expression E and L_pred the predictive loss.
                Automatically output by PySR and accessible in the Hall of Fame file.
    nmse_list:  taken from csv using "read_hof_csv"
    oos_nmse_list: optional, if you have an out-of-sample list of loss error  to compare against.
    save_path: directory for saving graph (optional)
    '''

    plt.figure(figsize=(8, 6))
    plt.xscale('log')
    plt.yscale('log')

    # Plot in-sample losses
    plt.scatter(complexity, nmse_list, label='In-Sample NMSE', color='green')
    plt.plot(complexity, nmse_list, color='green', alpha=0.5)

    # Plot out-of-sample losses if available
    if oos_nmse_list:
        plt.scatter(complexity, oos_nmse_list, label='Out-of-Sample NMSE', color='red')
        plt.plot(complexity, oos_nmse_list, color='red', alpha=0.5)

    # Highlight the "best" function if provided
    if best is not None and 0 <= best < len(complexity):
        plt.scatter(complexity[best], nmse_list[best], color='blue', s=100, zorder=5, label='Best Function')
        plt.annotate(f"Best",
                     (complexity[best], nmse_list[best]),
                     textcoords="offset points",
                     xytext=(10, 10),
                     ha='center',
                     fontsize=10,
                     arrowprops=dict(facecolor='blue', arrowstyle="->"))

    # Graph labels and styling
    plt.xlabel('Log Complexity')
    plt.ylabel('Log NMSE')
    plt.title('Log Complexity vs Log NMSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save or display the plot
    if save_path:
        plt.savefig(save_path + "_loglog")
    else:
        plt.show()
    plt.close()

def plot_complexity_vs_loss_loglog(complexity, loss_list, oos_loss_list=None, save_path=None,best=None):
    '''
    Parameters:
    -----------
    complexity: L(E) = L_pred(E) + (parsimony) · C(E), for C(E) the complexity of an expression E and L_pred the predictive loss.
                Automatically output by PySR and accessible in the Hall of Fame file.
    loss_list:  taken from csv using "read_hof_csv"
    oos_loss_list: optional, if you have an out-of-sample list of loss error  to compare against.
    save_path: directory for saving graph (optional)
    '''

    plt.figure(figsize=(8, 6))
    plt.xscale('log')
    plt.yscale('log')

    # Plot in-sample losses
    plt.scatter(complexity, loss_list, label='In-Sample Loss', color='green')
    plt.plot(complexity, loss_list, color='green', alpha=0.5)

    # Plot out-of-sample losses if available
    if oos_loss_list:
        plt.scatter(complexity, oos_loss_list, label='Out-of-Sample Loss', color='red')
        plt.plot(complexity, oos_loss_list, color='red', alpha=0.5)

    # Highlight the "best" function if provided
    if best is not None and 0 <= best < len(complexity):
        plt.scatter(complexity[best], loss_list[best], color='blue', s=100, zorder=5, label='Best Function')
        plt.annotate(f"Best",
                     (complexity[best], loss_list[best]),
                     textcoords="offset points",
                     xytext=(10, 10),
                     ha='center',
                     fontsize=10,
                     arrowprops=dict(facecolor='blue', arrowstyle="->"))

    # Graph labels and styling
    plt.xlabel('Log Complexity')
    plt.ylabel('Log Loss')
    plt.title('Log Complexity vs Log Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save or display the plot
    if save_path:
        plt.savefig(save_path + "_loglog")
    else:
        plt.show()
    plt.close()

def plot_complexity_vs_loss(complexity, loss_list, oos_loss_list=None, save_path=None, best=None):
    '''
    Parameters:
    -----------
    complexity: L(E) = L_pred(E) + (parsimony) · C(E), for C(E) the complexity of an expression E and L_pred the predictive loss.
                Automatically output by PySR and accessible in the Hall of Fame file.
    loss_list: taken from CSV using "read_hof_csv".
    oos_loss_list: Optional, if you have an out-of-sample list of loss errors to compare against.
    save_path: Directory for saving the graph (optional).
    best: Optional integer indicating the "best" function index to highlight in the graph.
    '''
    plt.figure(figsize=(8, 6))

    # Plot in-sample losses
    plt.scatter(complexity, loss_list, label='In-Sample Loss', color='green')
    plt.plot(complexity, loss_list, color='green', alpha=0.5)

    # Plot out-of-sample losses if available
    if oos_loss_list:
        plt.scatter(complexity, oos_loss_list, label='Out-of-Sample Loss', color='red')
        plt.plot(complexity, oos_loss_list, color='red', alpha=0.5)

    # Highlight the "best" function if provided
    if best is not None and 0 <= best < len(complexity):
        plt.scatter(complexity[best], loss_list[best], color='blue', s=100, zorder=5, label='Best Function')
        plt.annotate(f"Best",
                     (complexity[best], loss_list[best]),
                     textcoords="offset points",
                     xytext=(10, 10),
                     ha='center',
                     fontsize=10,
                     arrowprops=dict(facecolor='blue', arrowstyle="->"))

    # Graph labels and styling
    plt.xlabel('Complexity')
    plt.ylabel('Loss')
    plt.title('Complexity vs Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def read_hof_csv(hall_of_fame_csv, oos_csv=None):
    """
    Extract complexity, NMSE, function, and optionally OOS_NMSE from CSV files.
    
    Parameters:
    -----------
    hall_of_fame_csv : str
        Path to the hall of fame CSV containing Complexity and NMSE values.
    oos_csv : str, optional
        Path to the out-of-sample CSV containing OOS_NMSE values (default is None).
    
    Returns:
    --------
    tuple
        (complexity, loss_list, function_list, oos_nmse_list), where oos_nmse_list can be None.
    """
    try:
        # Load in-sample data (hall of fame)
        hof_data = pd.read_csv(hall_of_fame_csv)
        complexity = hof_data['Complexity'].values
        loss_list = hof_data['Loss'].values
        function_list = hof_data['Equation'].values

        # Load out-of-sample data if provided
        if oos_csv:
            oos_data = pd.read_csv(oos_csv)
            oos_loss_list = oos_data['Loss'].values
        else:
            oos_loss_list = None
        
        return complexity, loss_list, function_list, oos_loss_list

    except Exception as e:
        print(f"Error processing CSV files: {e}")
        return None, None, None, None

def find_csv(base_directory):
    for csv_folder in os.listdir(base_directory):

        # skip non-folders.
        if csv_folder.endswith(".csv"):
            print(f"Skipping invalid (or non folder): {csv_folder}")
            continue
        
        # create new filepath to open
        csv_path = os.path.join(base_directory, csv_folder)

        # troubleshooting opening filepath
        print(f"csv_path = {csv_path}")

        #open directory and search through it
        for csv_file in os.listdir(csv_folder):
            if csv_file.endswith(".csv"):
                print(f"csv found: {csv_file}")
                # creating new filepaths to open
                dataset_path = os.path.join(csv_folder, csv_file)
                print(f"dataset_path = {dataset_path}")
                return dataset_path