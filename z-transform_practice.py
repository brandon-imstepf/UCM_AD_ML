import lcapy as lc
from lcapy.discretetime import n
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import sympy as sp
from stat_plots import read_hof_csv, plot_complexity_vs_loss


# Exploring z-transformations (stack overflow)
#xk=n*2**n*lc.exp(3j*n)
#X0=xk.ZT()
#print(X0)

base_directory = 'C:/Users/brand/Desktop/Raj-Sindi/training_data/sim_csv_v6/11-14'
os.chdir(base_directory)

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
            # create elbow plot
            complexity, loss_list, function_list, oos_loss_list = read_hof_csv(dataset_path, oos_csv=None)
            plot_complexity_vs_loss(complexity, loss_list, oos_loss_list=None, save_path=csv_folder + "/")
            print(f"Saved plot in:  {csv_folder + "/"}")
