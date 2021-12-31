import numpy as np
import pandas as pd


def extracting_data() -> tuple:
    # Extracting the data from the .csv file
    data_frame = pd.read_csv(r'./toronto_rentals.csv')

    # Creating a numpy array containing the values of columns 'Bedroom', 'Bathroom', and 'Den'
    bbd_columns: pd.DataFrame = data_frame.iloc[:, :3]

    # Transforming the data Frame to a Numpy array
    bbd_columns: np.ndarray = bbd_columns.to_numpy()

    # Creating a data frame of rental prices values (as a float type)
    rental_prices: pd.DataFrame = data_frame.iloc[:, -1].str.replace(r'([\$,])', '', regex=True).astype('float')

    # Transforming the data Frame to a Numpy array
    rental_prices: np.ndarray = rental_prices.to_numpy()

    return bbd_columns, rental_prices
