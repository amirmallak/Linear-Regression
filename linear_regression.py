import numpy as np

from sklearn.linear_model import LinearRegression


def linear_reg_library_sol(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> tuple:
    print(f'\nFitting a Linear Regressor using Library solution...\n')

    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    y_pred = regressor.predict(x_test)

    loss_reg = np.mean(np.square(y_pred - y_test), axis=0)

    return loss_reg, regressor


def linear_reg_sol(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    print(f'\nFitting a Linear Regressor using a closed-form solution...\n')

    n = x.shape[0]  # Number of samples - N

    # Appending a column of 1's to x (to include the bias term)
    x = np.append(x, np.ones((n, 1)), axis=1)  # x shape - (N x D+1)

    # Calculating Linear Regression closed-form solution
    w = np.dot(np.linalg.inv(x.T @ x), (x.T @ y))  # w shape - (D+1 x 1)

    return w


def linear_reg_sol_via_gradient_descent(x: np.ndarray, y: np.ndarray, lr: float = 1e-4) -> np.ndarray:
    print(f'\nFitting a Linear Regressor using Gradient Descent solution...\n')
    n = x.shape[0]  # Number of samples - N

    # Appending a column of 1's to x (to include the bias term)
    x = np.append(x, np.ones((n, 1)), axis=1)  # x shape - (N x D+1)

    d = x.shape[1]  # Feature of each sample (the dimensionality). d size - D+1

    # Initializing our model
    w = np.ones((d, 1))  # w shape - (D+1 x 1)

    # epsilon = 970.272e-4  # The minimal loss after a few seconds
    loss = 1
    loss_history = []
    iteration = 0

    while iteration < 1e5:
        # Predicting the results using a linear regression model
        y_hyp = x @ w  # y_hyp shape - (N x 1)

        # Calculating the Loss function
        loss = (1 / (2 * n)) * ((y_hyp - y).T @ (y_hyp - y))  # Loss is a number
        loss = loss[0][0]
        if not iteration % 1e4:
            print(f'Loss = {loss}')

        # In case of fine tuning our solution for a slightly better performance
        # if loss <= 0.09702712:
        #     break
        # if iteration > 206e3:
        #     lr = 1e-5
        # if iteration > 5e5:
        #     lr = 1e-6

        loss_history.append(loss)

        # Calculating the derivative of the loss with respect to vector w
        df_dw = (1 / n) * (x.T @ (y_hyp - y))  # df_dw shape - (D+1 x 1)

        # Updating vector's w values (N x D) x (D x 1) = (N x 1)
        w -= lr * df_dw

        iteration += 1

    # Note: Could return also the Loss history and the number of iterations it took (in order to plot a Loss graph)
    return w
