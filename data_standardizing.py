import numpy as np

from sklearn.preprocessing import StandardScaler


def standardizing(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> tuple:
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]

    # Standardizing the data according to the training data
    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)

    y_train = y_train.reshape(n_train, 1)
    y_train = scaler.fit_transform(y_train)

    x_train_mean = np.mean(x_train)
    y_train_mean = np.mean(y_train)

    x_train_std = np.std(x_train)
    y_train_std = np.std(y_train)

    x_test = (x_test - x_train_mean) / x_train_std

    y_test = y_test.reshape(n_test, 1)
    y_test = (y_test - y_train_mean) / y_train_std

    return x_train, x_test, y_train, y_test
