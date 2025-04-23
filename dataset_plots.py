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