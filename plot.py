import numpy as np
import matplotlib.pyplot as plt

from typing import List


def plotting(loss_history: List[List]):
    print('\n--- Plotting ---')
    c_values = np.linspace(1e-1, 1, int(1e1))
    titles = ['Ridge Loss close-form on Training set',
              'Ridge Loss close-form on Test set',
              'Ridge Loss with GD on Training set',
              'Ridge Loss with GD on Test set',
              'Ridge Loss sklearn Regressor on Training set',
              'Ridge Loss sklearn Regressor on Test set']
    figure, axs = plt.subplots(3, 2)
    for row, ax in enumerate(axs):
        for col, a in enumerate(ax):
            a.plot(c_values, loss_history[2 * row + col])
            a.set_title(titles[2 * row + col])

    for ax in axs.flat:
        ax.set(xlabel='C values', ylabel='Loss')

    plt.show()
