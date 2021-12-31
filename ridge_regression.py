import numpy as np

from sklearn.preprocessing import StandardScaler


def ridge_regression_sol(x: np.ndarray, y: np.ndarray, c: float) -> np.ndarray:
    print(f'\nFitting a Ridge Regressor using a closed-form solution...\n')
    n = x.shape[0]  # Number of samples - N

    # Appending a column of 1's to x (to include the bias term)
    x = np.append(x, np.ones((n, 1)), axis=1)  # x shape - (N x D+1)

    d = x.shape[1]  # x dimensionality
    eye_matrix = np.eye(d)

    # Calculating Ridge Regression closed-form solution
    w = np.dot(np.linalg.inv((x.T @ x) + (c * eye_matrix)), (x.T @ y))  # w shape - (D+1 x 1)

    return w


def ridge_regression_sol_via_gradient_descent(x: np.ndarray, y: np.ndarray, c: float, lr: float = 1e-4) -> np.ndarray:
    print(f'\nFitting a Ridge Regressor using Gradient Descent solution...\n')
    n = x.shape[0]  # Number of samples - N

    # Appending a column of 1's to x (to include the bias term)
    x = np.append(x, np.ones((n, 1)), axis=1)  # x shape - (N x D+1)

    d = x.shape[1]  # Feature of each sample (the dimensionality). d size - D+1

    # Initializing our model
    w = np.ones((d, 1))  # w shape - (D+1 x 1)

    loss_history = []
    iteration = 0

    while iteration < 1e5:
        # Predicting the results using a linear regression model
        y_hyp = x @ w  # y_hyp shape - (N x 1)

        # Calculating the Loss function
        loss = (1 / n) * (((y_hyp - y).T @ (y_hyp - y)) + (c * np.sum(np.square(w[:-1]))))  # Loss is a number
        loss = loss[0][0]  # Getting a number from a 2d tensor

        if not iteration % 1e4:
            print(loss)

        loss_history.append(loss)

        # Calculating the derivative of the loss with respect to vector w
        # Calculating the derivative according to Ws and Bs with respect to the L2 Norm
        dc = (c * w[:-1]) / np.sqrt(np.sum(np.square(w[:-1])))  # dc shape - (D x 1)
        # dc = np.append(c * w[:-1], np.zeros(1)).reshape(d, 1)  # dc shape - (D+1 x 1)
        dc = np.append(dc, np.zeros(1)).reshape(d, 1)  # dc shape - (D+1 x 1)
        df_dw = ((x.T @ (y_hyp - y)) + dc)  # df_dw shape - (D+1 x 1)

        # Updating vector's w values
        w -= lr * df_dw

        iteration += 1

    # Note: Could return also the Loss history and the number of iterations it took (in order to plot a Loss graph)
    return w
