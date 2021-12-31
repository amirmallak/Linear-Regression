import numpy as np

from typing import List
from loss_metric import loss_function
from sklearn.linear_model import Ridge
from ridge_regression import ridge_regression_sol, ridge_regression_sol_via_gradient_descent


def ridge_reg_fit(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> List[List]:
    loss_history: List[List] = [[] for _ in range(6)]  # Creates a list of an empty 6 lists
    functions = [ridge_regression_sol, ridge_regression_sol_via_gradient_descent]
    x_data = [x_train, x_test]
    y_data = [y_train, y_test]

    print('\n--- Fitting Ridge Regression using Lambda Regularization term ---\n')
    for c in np.linspace(1e-1, 1, int(1e1)):
        print(f'\nLambda = {c}')
        w_data = []  # For saving the vector w of each fitting ridge function (end total size is 2)
        i, j = 0, 0

        for func in functions:
            for x, y in zip(x_data, y_data):
                if not i % 2:
                    w_data.append(func(x, y, c))

                loss = loss_function(x, y, w_data[i // 2])
                loss += c * np.sum(np.square(w_data[i // 2][:-1]))  # Adding the "L2 Norm" of w vector

                loss_history[i].append(loss)

                if not j % 2:
                    if not i % 2:
                        # Fit the data also by using Ridge sklearn Library Solution
                        ridge_regressor = Ridge(alpha=c)

                        ridge_regressor.fit(x, y)

                    y_predict = ridge_regressor.predict(x)

                    loss = np.mean(np.square(y_predict - y))

                    loss_history[4 + i].append(loss)

                i += 1
            j += 1

    return loss_history
