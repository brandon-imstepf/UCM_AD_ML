import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde


## Load file
# Load base directory
basedir = "C:/Users/brand/Documents/MATLAB/Raj_Torok Lab/sim_csv_v5/Sept 16/"
# Load individual file
csv_file = basedir + "master_flux_data.csv"

# Convert file to pandas dataframe (so we can extract the Nrow and Ncol)
csv_df = pd.read_csv(csv_file)

# Get NRow and NCol data from dataframe
Nrow_data = csv_df['NRow']
Ncol_data = csv_df['NCol']

# Create histograms for Nrow and Ncol
plt.figure(figsize=(12, 6))

# Formatter for scientific notation
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-3, 3))
#################

## Make lists to sort maximum and minimum values
#column stuff
col_list = Ncol_data.to_list()
col_list.sort(reverse=True)
ncolmax = col_list[:5]

col_list_nozero = [x for x in Ncol_data.to_list() if x != 0]
col_list_nozero.sort(reverse=True)
ncolmin = col_list_nozero[-5:]

# row stuff
row_list = Nrow_data.to_list() 
row_list.sort(reverse=True)
nrowmax = row_list[:5]

row_list_nozero = [x for x in Nrow_data.to_list() if x != 0]
row_list_nozero.sort(reverse=True)
nrowmin = row_list_nozero[-5:]

print(f"Maximum and minimum values for NCol = {ncolmax,ncolmin}")
print(f"Maximum and minimum values for NRow = {nrowmax,nrowmin}")

# Can make distribution out of minimum and maximum values.
# Can the weird 1e-16 values be left out of distribution?

# Histogram for Nrow
plt.subplot(2, 2, 1)
plt.hist(Nrow_data, bins=100, color='blue', alpha=0.7)
plt.title('Histogram of NRow')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Apply scientific notation to x and y axes
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)

# Histogram for Ncol
plt.subplot(2, 2, 2)
plt.hist(Ncol_data, bins=100, color='green', alpha=0.7)
plt.title('Histogram of NCol')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Apply scientific notation to x and y axes
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)

# Histogram for NRow+Ncol
plt.subplot(2, 2, 3)
plt.hist2d(Ncol_data, Ncol_data, bins=20, color='red', alpha=0.7)
plt.title('Histogram of NCol NRow')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Apply scientific notation to x and y axes
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)


# Density scatterplot for Nrow+Ncol
xy = np.vstack([Ncol_data,Nrow_data])
z = gaussian_kde(xy)(xy)
plt.subplot(2, 2, 4)
plt.scatter(Nrow_data,Ncol_data,c=z,s=100)
#plt.hist2d(nrowmax, ncolmax, bins=10, color='black', alpha=0.7)
plt.title('Gaussian KDE of NCol & NCol')
plt.xlabel('NRow')
plt.ylabel('NCol')

# Apply scientific notation to x and y axes
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)


# Show the plot
plt.tight_layout()
plt.show()



