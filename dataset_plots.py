# dataset plots

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_line_accuracy(df):
    """ Line Plot: Accuracy vs. Noise Level for each dataset """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Noise Level', y='Accuracy (%)', hue='Dataset', marker='o')
    plt.xticks(rotation=45)
    plt.title('Accuracy vs. Noise Level')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Noise Level')
    plt.legend(title='Dataset', bbox_to_anchor=(1, 1))
    plt.grid()
    plt.show()

def plot_accuracy_vs_noise_2(results_df):
    """Scatter plot of accuracy vs. noise level."""
    best_accuracies = results_df.groupby(['Dataset', 'Noise Level'])['Accuracy (%)'].max().reset_index()
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=best_accuracies, x='Noise Level', y='Accuracy (%)', hue='Dataset', s=100, edgecolor='black', alpha=0.8)
    plt.xticks(rotation=45, ha='right')
    plt.title('Accuracy vs. Noise Level')
    plt.xlabel('Noise Level')
    plt.ylabel('Accuracy (%)')
    plt.legend(title='Dataset', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.show()

def plot_bar_best_accuracy(df):
    """ Bar Chart: Best Accuracy per Noise Level """
    best_acc_df = df.groupby(['Noise Level'])['Accuracy (%)'].max().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=best_acc_df, x='Noise Level', y='Accuracy (%)', palette='viridis')
    plt.xticks(rotation=45)
    plt.title('Best Accuracy per Noise Level')
    plt.ylabel('Best Accuracy (%)')
    plt.xlabel('Noise Level')
    plt.grid()
    plt.show()

def plot_heatmap_accuracy(df):
    """ Heatmap: Accuracy vs. Dataset & Noise Level """
    pivot_df = df.pivot(index='Dataset', columns='Noise Level', values='Accuracy (%)')
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='coolwarm', linewidths=0.5)
    plt.title('Heatmap of Accuracy by Dataset and Noise Level')
    plt.ylabel('Dataset')
    plt.xlabel('Noise Level')
    plt.xticks(rotation=45)
    plt.show()

def plot_scatter_complexity_accuracy(df):
    """ Scatter Plot: Complexity vs. Accuracy """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Complexity', y='Accuracy (%)', hue='Noise Level', size='Loss', palette='coolwarm', sizes=(20, 200))
    plt.title('Complexity vs. Accuracy')
    plt.xlabel('Equation Complexity')
    plt.ylabel('Accuracy (%)')
    plt.grid()
    plt.legend(title='Noise Level', bbox_to_anchor=(1, 1))
    plt.show()

def summarize_best_equations(df):
    """ Table Summary: Best Equations Found """
    best_equations = df.loc[df.groupby('Dataset')['Accuracy (%)'].idxmax(), ['Dataset', 'Noise Level', 'Matched Equation', 'Accuracy (%)']]
    print(best_equations.to_string(index=False))


def plot_accuracy_vs_noise(results_df):
    """Line plot of accuracy vs. noise level."""
    best_accuracies = results_df.groupby(['Dataset', 'Noise Level'])['Accuracy (%)'].max().reset_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=best_accuracies, x='Noise Level', y='Accuracy (%)', hue='Dataset', marker='o')
    plt.title('Accuracy vs. Noise Level')
    plt.xlabel('Noise Level')
    plt.ylabel('Accuracy (%)')
    plt.legend(title='Dataset')
    plt.grid(True)
    plt.show()

def plot_best_accuracy_per_noise(results_df):
    """Bar chart of the best accuracy per noise level."""
    best_accuracies = results_df.groupby('Noise Level')['Accuracy (%)'].max().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=best_accuracies, x='Noise Level', y='Accuracy (%)', palette='viridis')
    plt.title('Best Accuracy per Noise Level')
    plt.xlabel('Noise Level')
    plt.ylabel('Best Accuracy (%)')
    plt.grid(axis='y')
    plt.show()

def plot_accuracy_heatmap(results_df):
    """Heatmap of accuracy vs. dataset and noise level."""
    pivot_table = results_df.pivot_table(values='Accuracy (%)', index='Dataset', columns='Noise Level', aggfunc='max')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, cmap='coolwarm', fmt='.1f')
    plt.title('Accuracy Heatmap')
    plt.xlabel('Noise Level')
    plt.ylabel('Dataset')
    plt.show()

def plot_complexity_vs_accuracy(results_df):
    """Scatter plot of equation complexity vs. accuracy."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=results_df, x='Complexity', y='Accuracy (%)', hue='Dataset', alpha=0.7)
    plt.title('Complexity vs. Accuracy')
    plt.xlabel('Equation Complexity')
    plt.ylabel('Accuracy (%)')
    plt.legend(title='Dataset')
    plt.grid(True)
    plt.show()

def plot_best_accuracy_table(results_df):
    """Table of the best equations found per dataset."""
    best_equations = results_df.loc[results_df.groupby('Dataset')['Accuracy (%)'].idxmax()][['Dataset', 'Noise Level', 'Matched Equation', 'Complexity', 'Accuracy (%)']]
    print(best_equations.to_string(index=False))

def plot_best_accuracy_per_function(results_df):
    """Bar plot of highest recovery percentage per function at different noise levels."""
    best_accuracies = results_df.groupby(['Dataset', 'Noise Level'])['Accuracy (%)'].max().reset_index()
    noise_levels = sorted(results_df['Noise Level'].unique())
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=best_accuracies, x='Dataset', y='Noise Level', size='Accuracy (%)', hue='Accuracy (%)', palette='coolwarm', sizes=(20, 200))
    plt.xticks(rotation=45, ha='right')
    plt.title('Highest Accuracy per Function at Different Noise Levels')
    plt.xlabel('Function Name')
    plt.ylabel('Noise Level')
    plt.grid(True)
    plt.show()

def plot_accuracy_heatmap_per_function(results_df):
    """Heatmap of accuracy per function across different noise levels."""
    pivot_table = results_df.pivot_table(values='Accuracy (%)', index='Dataset', columns='Noise Level', aggfunc='max')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, cmap='RdYlGn', fmt='.1f', linewidths=0.5, linecolor='black')
    plt.title('Accuracy Heatmap per Function')
    plt.xlabel('Noise Level')
    plt.ylabel('Function Name')
    plt.show()




def plot_accuracy_vs_noise3(results_df):
    """Line plot of accuracy vs. noise level."""
    best_accuracies = results_df.groupby(['Dataset', 'Noise Level'])['Accuracy (%)'].max().reset_index()
    
    # Truncate noise level names
    best_accuracies['Truncated Noise Level'] = best_accuracies['Noise Level'].apply(lambda x: x.split('_')[0])
    
    # Convert to numeric and sort by truncated noise level
    best_accuracies['Truncated Noise Level'] = pd.Categorical(best_accuracies['Truncated Noise Level'], categories=['0', '1', '3', '5', '10', '25', '50'], ordered=True)
    best_accuracies = best_accuracies.sort_values(by='Truncated Noise Level')
    
    plt.figure(figsize=(10, 6))
    
    for dataset in best_accuracies['Dataset'].unique():
        subset = best_accuracies[best_accuracies['Dataset'] == dataset]
        plt.plot(subset['Truncated Noise Level'], subset['Accuracy (%)'], marker='o', label=dataset)
    
    plt.title('Accuracy vs. Noise Level')
    plt.xlabel('Noise Level')
    plt.ylabel('Accuracy (%)')
    plt.legend(title='Dataset')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

def plot_variable_density(X, feature_columns, output_dir=None, filename_prefix="variable_density"):
    """
    Plots a variable density (KDE) for each feature in X.
    Args:
        X: 2D numpy array of features
        feature_columns: list or array of feature names
        output_dir: directory to save plots (optional)
        filename_prefix: prefix for saved plot files
    """
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    X = np.array(X)
    n_features = X.shape[1]
    for i in range(n_features):
        plt.figure(figsize=(8, 5))
        sns.kdeplot(X[:, i], fill=True, color='blue', alpha=0.5)
        plt.title(f"Density Plot for {feature_columns[i]}")
        plt.xlabel(feature_columns[i])
        plt.ylabel("Density")
        plt.grid(True, ls='--', alpha=0.5)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"{filename_prefix}_{feature_columns[i]}.png"))
            plt.close()
        else:
            plt.show()

def advanced_error_analysis(y_true, y_pred, feature_matrix=None, feature_names=None, output_dir=None, prefix=""): 
    """
    Advanced error analysis: residual plots, QQ plot, error vs. features.
    Args:
        y_true: true target values
        y_pred: predicted target values
        feature_matrix: (optional) features for error vs. feature plots
        feature_names: (optional) names of features
        output_dir: (optional) directory to save plots
        prefix: (optional) prefix for plot filenames
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.stats as stats
    import os
    residuals = y_true.flatten() - y_pred.flatten()
    # Residual vs. Predicted
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Predicted')
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{prefix}residuals_vs_pred.png"))
        plt.close()
    else:
        plt.show()
    # QQ plot
    plt.figure()
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('QQ Plot of Residuals')
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"{prefix}qqplot.png"))
        plt.close()
    else:
        plt.show()
    # Error vs. Features
    if feature_matrix is not None and feature_names is not None:
        for i, name in enumerate(feature_names):
            plt.figure()
            plt.scatter(feature_matrix[:, i], residuals, alpha=0.5)
            plt.xlabel(name)
            plt.ylabel('Residuals')
            plt.title(f'Residuals vs. {name}')
            if output_dir:
                plt.savefig(os.path.join(output_dir, f"{prefix}residuals_vs_{name}.png"))
                plt.close()
            else:
                plt.show()

def plot_linear_regression_residuals(y_train=None, y_train_pred=None, y_test=None, y_test_pred=None, output_dir=None):
    """
    Move or plot linear regression residuals to the Visualizations directory.
    If predictions are not provided, tries to load them from CSVs in output_dir.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    if y_train_pred is None and output_dir is not None:
        try:
            y_train_pred = pd.read_csv(os.path.join(output_dir, "linear_train_pred.csv")).values.flatten()
            y_test_pred = pd.read_csv(os.path.join(output_dir, "linear_test_pred.csv")).values.flatten()
        except Exception as e:
            print(f"Could not load linear regression predictions: {e}")
            return
    if y_train is None or y_test is None or y_train_pred is None or y_test_pred is None:
        print("Insufficient data for plotting linear regression residuals.")
        return
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
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "linear_regression_residuals.png"))
    plt.close()


def compare_regression_metrics(equation_dir, vis_dir):
    """
    Compare symbolic regression and linear regression metrics visually.
    Reads best.txt and linear_regression_results.txt, and plots bar charts for MSE, NMSE, RMSE.
    """
    import os
    import matplotlib.pyplot as plt
    import re
    # Read symbolic regression metrics
    best_path = os.path.join(equation_dir, "best.txt")
    linreg_path = os.path.join(equation_dir, "linear_regression_results.txt")
    metrics = [
        ("Train MSE", "Train MSE"),
        ("Test MSE", "Test MSE"),
        ("Validation MSE", "Validation MSE"),
        ("Train NMSE", "Train NMSE"),
        ("Test NMSE", "Test NMSE"),
        ("Validation NMSE", "Validation NMSE"),
        ("Train RMSE", "Train RMSE"),
        ("Test RMSE", "Test RMSE"),
        ("Validation RMSE", "Validation RMSE")
    ]
    sym_vals = {}
    lin_vals = {}
    def extract_metrics(path, dest):
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return
        with open(path, "r") as f:
            for line in f:
                for m, _ in metrics:
                    if m in line:
                        val = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", line)
                        if val:
                            dest[m] = float(val[0])
    extract_metrics(best_path, sym_vals)
    extract_metrics(linreg_path, lin_vals)
    os.makedirs(vis_dir, exist_ok=True)
    for m, label in metrics:
        if m in sym_vals and m in lin_vals:
            plt.figure()
            plt.bar(["Symbolic", "Linear"], [sym_vals[m], lin_vals[m]], color=["orange", "blue"])
            plt.title(f"{label} Comparison")
            plt.ylabel(label)
            plt.savefig(os.path.join(vis_dir, f"compare_{label.replace(' ', '_').lower()}.png"))
            plt.close()
        else:
            print(f"Metric {m} missing in one or both result files.")

