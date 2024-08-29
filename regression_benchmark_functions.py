# regression_benchmark_functions
import numpy as np

# Define Nguyen functions
def Nguyen_F1(x): return x**3 + x**2 + x
def Nguyen_F2(x): return x**4 + x**3 + x**2 + x
def Nguyen_F3(x): return x**5 + x**4 + x**3 + x**2 + x
def Nguyen_F4(x): return x**6 + x**5 + x**4 + x**3 + x**2 + x
def Nguyen_F5(x): return np.sin(x**2) * np.cos(x) - 1
def Nguyen_F6(x): return np.sin(x) + np.sin(x + x**2)
def Nguyen_F7(x): return np.log(x + 1) + np.log(x**2 + 1)
def Nguyen_F8(x): return np.sqrt(x)
def Nguyen_F9(x, y): return np.sin(x) + np.sin(y**2)
def Nguyen_F10(x, y): return 2 * np.sin(x) * np.cos(y)

Nguyen_funcs = [Nguyen_F1, Nguyen_F2, Nguyen_F3, Nguyen_F4, Nguyen_F5, 
                Nguyen_F6, Nguyen_F7, Nguyen_F8, Nguyen_F9, Nguyen_F10]

# Define Korns functions
def Korns_1(X3): return 1.57 + (24.3 * X3)
def Korns_2(X1, X3, X4): return 0.23 + (14.2 * ((X3 + X1) / (3.0 * X4)))
def Korns_3(X0, X1, X3, X4): return -5.41 + (4.9 * (((X3 - X0) + (X1 / X4)) / (3 * X4)))
def Korns_4(X2): return -2.3 + (0.13 * np.sin(X2))
def Korns_5(X4): return 3.0 + (2.13 * np.log(X4))
def Korns_6(X0): return 1.3 + (0.13 * np.sqrt(X0))
def Korns_7(X0): return 213.80940889 - (213.80940889 * np.exp(-0.54723748542 * X0))
def Korns_8(X0, X3, X4): return 6.87 + (11 * np.sqrt(7.23 * X0 * X3 * X4))
def Korns_9(X0, X1, X2, X3): return ((np.sqrt(X0) / np.log(X1)) * (np.exp(X2) / np.square(X3)))
def Korns_10(X1, X2, X3, X4): return 0.81 + (24.3 * (((2.0 * X1) + (3.0 * np.square(X2))) / ((4.0 * np.power(X3, 3)) + (5.0 * np.power(X4, 4)))))
def Korns_11(X0): return 6.87 + (11 * np.cos(7.23 * X0 * X0 * X0))
def Korns_12(X0, X4): return 2.0 - (2.1 * (np.cos(9.8 * X0) * np.sin(1.3 * X4)))
def Korns_13(X0, X1, X2, X3): return 32.0 - (3.0 * ((np.tan(X0) / np.tan(X1)) * (np.tan(X2) / np.tan(X3))))
def Korns_14(X0, X1, X2, X3): return 22.0 + (4.2 * ((np.cos(X0) - np.tan(X1)) * (np.tanh(X2) / np.sin(X3))))
def Korns_15(X0, X1, X2, X3): return 12.0 - (6.0 * ((np.tan(X0) / np.exp(X1)) * (np.log(X2) - np.tan(X3))))
