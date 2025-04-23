import numpy as np
import matplotlib.pyplot as plt
from regression_benchmark_functions import *

orig_dataset_location = 'C:/Users/brand/Documents/Python Scripts/pysr_code/datasets/Nguyen-RBFs-n-ns-(1,3,40)/LEARNED/10_Percent_Noise/'
orig_dataset_filename = 'Nguyen_F5_LHS.csv'

# Sort Nguyen_F5_LHS dataset in ascending order
dataset = np.genfromtxt(orig_dataset_location + orig_dataset_filename, delimiter=',', skip_header=1)
dataset = dataset[dataset[:,0].argsort()]

# Generate 40 equally spaced points between 1 and 3
x = np.linspace(1, 3, 40)

# Evaluate the function at these points
y = Nguyen_F5(x)

# Plot the orig dataset
plt.scatter(dataset[:,0], dataset[:,1],color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset, 10% Noise')
plt.legend()
plt.show()

# Plot the function
plt.scatter(dataset[:,0], dataset[:,1], label='10% Noise',color='red')
plt.plot(x, y, label='sin(x^2)*cos(x) - 1')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Learned Function')
plt.legend()
plt.show()