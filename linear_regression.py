import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Function to perform linear regression
def perform_linear_regression(csv_file):
    # Load the data from CSV file
    data = pd.read_csv(csv_file)

    # Separate the features (all columns except the last one) and the target (last column)
    X = data.iloc[:, :-1]  # Features (all columns except the last)
    y = data.iloc[:, -1]   # Target (the last column)

    # Split the data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the linear regression model
    model = LinearRegression()

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Predict the target on the testing data
    y_pred = model.predict(X_test)

    # Calculate the mean squared error (MSE) on the test data
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Optionally, print the coefficients of the regression model
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)

    return model

# Specify the path to your CSV file
basedir = "C:/Users/brand/Documents/Python Scripts/pysr_code/datasets/Sept_20_Cumulative/"
filename = 'allparams_training.2928.csv'

# Perform linear regression on the CSV file
model = perform_linear_regression(basedir + filename)
