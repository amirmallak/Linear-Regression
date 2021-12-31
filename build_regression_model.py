from plot import plotting
from explanation import result_explaining
from data_extraction import extracting_data
from data_standardizing import standardizing
from ridge_regression_fitting import ridge_reg_fit
from sklearn.model_selection import train_test_split
from linear_regression_fitting import linear_reg_fit


def regression_model():
    # Extract the data from our rental dataset
    bbd_columns, rental_prices = extracting_data()

    # Split the data to training and test according to a 20% ratio
    x_train, x_test, y_train, y_test = train_test_split(bbd_columns, rental_prices, test_size=0.2, random_state=42)

    x_train, x_test, y_train, y_test = standardizing(x_train, x_test, y_train, y_test)

    # Fitting our data with various Linear Regression methods and printing the results
    linear_reg_fit(x_train, x_test, y_train, y_test)

    # Fitting our data with different Ridge Regression methods using different lambdas (regularization term)
    loss_history = ridge_reg_fit(x_train, x_test, y_train, y_test)

    # Plotting the Loss graphs of Ridge Regression model with relation to the different lambdas used
    plotting(loss_history)

    result_explaining()
