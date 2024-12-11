import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import sympy as sp
from sklearn.model_selection import train_test_split

# Need to open .csv, randomly take n rows of data (testing), then take m% of those rows (validation)
# sklearn has built in test/train split ... but I can also do stuff with that.
# need to generate a folder with this data in it to be manipulated further

def train_validate_split(test_size_input,csv):
    # troubleshooting
    print("You have called test_validate_split function.")

    # reading data
    csv_data = csv
    #csv_data = pd.read_csv(csv)

    # selecting test_rows number of rows randomly
    print("Train / Validate split set at: 80/20")
    train_data, validation_data = train_test_split(csv_data, test_size=test_size_input)


data = range(1,10)

train_data, validation_data = train_validate_split(0.2,data)

print(train_data)
print('---------')
print(validation_data)


    

    