import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

# 2b. Define input ranges
gamma1_range = [5e-5, 5e-2]
lambda_range = [0.01, 0.1]
delta_range = [10, 100]
epsilon_range = [10, 100]
tau_x0_range = [0, 1.6e-4]
tau_xL_range = [0, 1.6e-4]

# Define number of iterations
numiterations = 10000  # You can adjust this value as needed

# 2c. Generate random samples from either log- or lin-spaced distributions
def logdist_fun(x, n):
    return 10 ** (np.log10(x[0]) + (np.log10(x[1]/x[0]) * np.random.rand(n)))

def lindist_fun(x, n):
    return x[0] + (x[1] - x[0]) * np.random.rand(n)

gamma1_vals = logdist_fun(gamma1_range, numiterations)
lambda_vals = logdist_fun(lambda_range, numiterations)
delta_vals = lindist_fun(delta_range, numiterations)
epsilon_vals = lindist_fun(epsilon_range, numiterations)
tau_x0_vals = lindist_fun(tau_x0_range, numiterations)
tau_xL_vals = lindist_fun(tau_xL_range, numiterations)

# Create histograms
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distributions of Generated Values', fontsize=16)

axs[0, 0].hist(gamma1_vals, bins=50, edgecolor='black')
axs[0, 0].set_title('gamma1 distribution')
axs[0, 0].set_xscale('log')

axs[0, 1].hist(lambda_vals, bins=50, edgecolor='black')
axs[0, 1].set_title('lambda distribution')
axs[0, 1].set_xscale('log')

axs[0, 2].hist(delta_vals, bins=50, edgecolor='black')
axs[0, 2].set_title('delta distribution')

axs[1, 0].hist(epsilon_vals, bins=50, edgecolor='black')
axs[1, 0].set_title('epsilon distribution')

axs[1, 1].hist(tau_x0_vals, bins=50, edgecolor='black')
axs[1, 1].set_title('tau_x0 distribution')

axs[1, 2].hist(tau_xL_vals, bins=50, edgecolor='black')
axs[1, 2].set_title('tau_xL distribution')

plt.tight_layout()

# Add axis labels and title
plt.xlabel('Value')
plt.ylabel('Distribution')

# Round x-axis tick labels to one decimal place
#plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

# Show the plot
plt.show()