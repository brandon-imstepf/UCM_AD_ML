import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import sympy as sp
from sympy import exp, sympify, sin, cos, log
from regression_benchmark_functions import Imstepf_funcs,Nguyen_funcs
import sklearn as sk
from sklearn import metrics
from sklearn.linear_model import LinearRegression
'''
start wit hthe first csv in the directory then look into subdirectories until folder found with same name (till second '_')
and extract from that


-- first, let's hardcode this using the "true" and the dataset csv and the pysr csv -- 

to do (10/25/24):
* Have code run through entire directory
* Label things modularly
    * Equation we're predicting
    * Predicted equation # (which line, which complexity)
'''

# MATPLOTLIB FONT SETTINGS #
from matplotlib.font_manager import FontProperties
font_prop = FontProperties(size=26, family='serif', style='normal')
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 14})


def getErrors(yTrue,yPred):
    mse = sk.metrics.mean_squared_error(yTrue,yPred)
    var_yTrue = np.var(yTrue)
    nmse = mse / var_yTrue
    return mse,nmse

def getResiduals(yTrue,yPred):
    residuals = abs(yTrue - yPred)
    return residuals

def getTroubleshoot(eq,xTrue,yTrue,yPred,residuals,mse,nmse):
    print(f'Chosen Equation = {eq}')
    print(f'xTrue = {xTrue}')
    print(f'yTrue = {yTrue}')
    print(f'yPred = {yPred}')
    print(f'Residuals = {residuals}')
    print(f'Mean Squared Error = {mse}')
    print(f'Normalized MSE = {nmse}')



def getSinglePlot(xTrue, yTrue, yPred, equation_str, complexity, nmse, mse):
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot the true function
    plt.plot(xTrue, yTrue, label='True Function', color='blue', linewidth=2)
    
    # Plot the predicted function from the hall of fame CSV
    plt.plot(xTrue, yPred, label='Predicted Function', linestyle='--', color='red')
    
    # Set labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Comparison of True Function vs Predicted Function \n {eq_str}')
    plt.legend()

    # Create the text box with additional info
    textstr = f"Equation: {equation_str}\nComplexity: {complexity}\nNMSE: {nmse:.4f}\nMSE: {mse:.4f}"
    
    # Position the text box on the right side of the plot
    plt.gcf().text(0.135, 0.66, textstr, fontsize=10, va='bottom', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    
    # Show/save the plot
    #plt.show()
    plt.savefig(f"complexity: {complexity} plot")


def getComplexityErrorPlot(complexity,nmse_list,oos_nmse_list):
    # Convert complexity to 1D array
    complexity = temp_complexity.flatten()

    # Complexity vs Error plot
    plt.figure(figsize=(8, 6))
    # in-sample
    plt.scatter(complexity, nmse_list, color='green')
    plt.plot(complexity, nmse_list, label='In-Sample', color='green', alpha=0.5)
    #out of sample
    plt.scatter(complexity,oos_nmse_list,label='Out-Of-Sample',color='red')
    plt.plot(complexity, oos_nmse_list, color='red', alpha=0.5)

    # Set labels and title
    plt.xlabel('Complexity')
    plt.ylabel('Normalized Mean Squared Error (NMSE)')
    plt.title('Complexity vs Error')

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()

#basedir = "C:/Users/brand/Documents/Python Scripts/pysr_code/datasets/Imstepf_RBFs-oos-(0,1,20)"
basedir = "C:/Users/brand/Documents/Python Scripts/pysr_code/datasets/RBFs-n-oos-(-1,1,20)"

os.chdir(basedir)

# For Nguyen F1 Latin Hypercube Sampling
trueDomain = pd.read_csv("Nguyen_F1_LHS.csv")
oosDomain = pd.read_csv("Nguyen_F1_LHS_(1,2,20).csv")
# get correct domains (in sequential order because LHS randomly samples)
xTrue = np.sort(trueDomain.iloc[:,0].values) # right now we're just taking first column, but eventually we'll need to take all but last
yTrue = np.sort(trueDomain.iloc[:,1].values) # the true range
# get correct domains (in sequential order because LHS OUT OF SAMPLE randomly samples)
xoosTrue = np.sort(oosDomain.iloc[:,0].values) # right now we're just taking first column, but eventually we'll need to take all but last
yoosTrue = np.sort(oosDomain.iloc[:,1].values) # the true range


#pysrCSV = pd.read_csv("Sept_30/Imstepf_F1_LHS/hall_of_fame_2024-09-30_140122.490.csv")
pysrCSV = pd.read_csv("oct_7_binary_only/Nguyen_F1_LHS/hall_of_fame_2024-10-07_105717.279.csv")

equations = pysrCSV['Equation'].values # list the equations extracted
eq_str = pysrCSV['Equation'].iloc[0]
temp_complexity = pysrCSV['Complexity'].values # get the complexity (to compare against error later)
# Convert complexity to 1D array
complexity = temp_complexity.flatten()

# define a sympy variable for symbolic manipualation
x0 = sp.symbols('x0')
x1 = sp.symbols('x1')
x2 = sp.symbols('x2')
x3 = sp.symbols('x3')
x4 = sp.symbols('x4')
x5 = sp.symbols('x5')
x6 = sp.symbols('x6')

# Convert the first equation string into a SymPy expression
eq = sp.sympify(eq_str)

# 3. Evaluate the first equation on the true domain values
yPred = [eq.subs(x0, val) for val in xTrue]
yoosPred = [eq.subs(x0, val) for val in xoosTrue]



residuals = getResiduals(yTrue,yPred)

# get errors
nmse_list = []
mse_list = []
oos_nmse_list =[]
oos_mse_list = []
for eq_str,comp in zip(equations,complexity):
    eq = sp.sympify(eq_str, locals={'exp': exp})
    yPred = [eq.subs(x0, val).evalf() for val in xTrue]
    yoosPred = [eq.subs(x0, val).evalf() for val in xoosTrue]
    mse, nmse = getErrors(yTrue, yPred)
    nmse_list.append(nmse)
    mse_list.append(mse)
    oos_mse, oos_nmse = getErrors(yoosTrue, yoosPred)
    oos_nmse_list.append(oos_nmse)
    oos_mse_list.append(oos_mse)

    getSinglePlot(xTrue,yTrue,yPred,eq_str,comp,nmse,mse)
    #getTroubleshoot(eq,xTrue,yTrue,yPred,residuals,mse,nmse)
    

#### PLOTTING ####

getComplexityErrorPlot(complexity,nmse_list,oos_nmse_list)