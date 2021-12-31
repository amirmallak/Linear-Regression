import numpy as np

from loss_metric import loss_function
from linear_regression import linear_reg_sol, linear_reg_sol_via_gradient_descent, linear_reg_library_sol


def linear_reg_fit(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
    print('\n--- Fitting the model using Linear Regression ---\n')

    # --- Fitting the model according to the different methods and calculating the Loss of the model for each method
    fit_model_func = [0, 0]  # Initial assignment
    w_model = []
    loss_model = []
    fit_model_func[0], fit_model_func[1] = linear_reg_sol, linear_reg_sol_via_gradient_descent

    for fitting_func in fit_model_func:
        # Fitting a model using the closed-form solution of Linear Regression
        w = fitting_func(x_train, y_train)
        w_model.append(w)

        # Calculating the Loss of the Linear Regression closed-form models
        loss = loss_function(x_test, y_test, w)
        loss_model.append(loss)

    print(f'\nThe Expected Loss value of the Linear Regression closed-form solution is: {loss_model[0]}')
    print(f'The Expected Loss value of the Linear Regression via Gradient Decent solution is: {loss_model[1]}\n')

    # Fitting a solution for our dataset using a Library Linear Regression Model
    loss_reg, regressor = linear_reg_library_sol(x_train, x_test, y_train, y_test)

    print(f'Matching to a Linear Regression Library Model:\n')
    print(f'Loss of Regressor is: {loss_reg / 1e9}e+9')
    print(f'The difference between Regressor Loss and closed-form Loss is: {np.abs(loss_reg - loss_model[0])}')
    print(f'The difference between Regressor Loss and our Loss is: {np.abs(loss_reg - loss_model[1])}')
    print(f'Difference between W vectors in the closed-form solution with respect to the sklearn\'s is: '
          f'{np.abs(regressor.coef_.reshape(3, 1) - w_model[0][:3])}')
    print(f'Difference between W vectors in our solution with respect to the sklearn\'s is: '
          f'{np.abs(regressor.coef_.reshape(3, 1) - w_model[1][:3])}\n')
