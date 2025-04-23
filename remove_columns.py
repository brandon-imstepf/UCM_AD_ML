import pandas as pd

def remove_second_to_last_column(input_file, output_file):
    """
    Remove the second-to-last column from a CSV file and save to a new file.

    Parameters:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the modified CSV file.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)
    print("Initial DataFrame shape:", df.shape)

    # Drop the second-to-last column
    if len(df.columns) > 1:
        df = df.drop(df.columns[-1], axis=1)
        print("Columns after dropping:", df.columns)

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print("Modified DataFrame saved to:", output_file)

# Example usage
base_directory = 'C:/Users/brand/Desktop/Raj-Sindi/training_data/sim_csv_v9_fixed/'
dataset_name = 'data_frac_07_widerange_all.csv'
input_csv = base_directory + dataset_name
output_csv = base_directory + 'flux_only_' + dataset_name

print("Input CSV:", input_csv)
print("Output CSV:", output_csv)
remove_second_to_last_column(input_csv, output_csv)
