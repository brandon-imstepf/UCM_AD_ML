def plot_loglog_complexity_vs_error(complexity, nmse_list, oos_nmse_list, save_path):
    """
    Create a log-log plot of error vs. complexity.
    Filters out invalid values to prevent overflow or plotting issues.
    """
    # Convert lists to NumPy arrays for easier filtering
    complexity = np.array(complexity)
    nmse_list = np.array(nmse_list)
    oos_nmse_list = np.array(oos_nmse_list)
    
    # Filter out invalid or zero values (logarithms undefined for these)
    valid_indices = (complexity > 0) & (nmse_list > 0) & (oos_nmse_list > 0)
    if not np.any(valid_indices):
        print("No valid data points for log-log plot.")
        return  # Skip plotting if no valid data

    # Filter the data
    complexity = complexity[valid_indices]
    nmse_list = nmse_list[valid_indices]
    oos_nmse_list = oos_nmse_list[valid_indices]

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.loglog(complexity, nmse_list, label='NMSE (In-Sample)', marker='o')
    plt.loglog(complexity, oos_nmse_list, label='NMSE (Out-of-Sample)', marker='s')
    plt.xlabel("Complexity (log scale)")
    plt.ylabel("Error (log scale)")
    plt.title("Error vs. Complexity (Log-Log Scale)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Save the plot
    plt.savefig(save_path)
    plt.close()