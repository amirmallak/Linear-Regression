
def result_explaining():
    print('\n --- Explaining the Ploted Results ---\n')
    print('We can see that the Loss, in all the training set graph graphs: closed-form solution, '
          'Gradient Descent method, and the sklearn linear ridge regressor, is increasing when the lambda term - λ (c) '
          'is increasing! (correlated)')
    print('As λ increases, the flexibility of the ridge regression fit decreases, leading to decreased variance but '
          'increased bias (Hence increasing Loss). If the co-efficients of predictors (Ws) decrease, then their value '
          'in the model decreases. That is, their effect decreases. And thus the flexibility of the model should '
          'decrease.')
    print('In the Linear Regression problem, limiting the weights values can lead to miss predicting (we can\'t really'
          'overfit our data - it\'s a linear model!) in the ambient space (in our case 3d dimension).')
    print('When in one axis (or eigenvector representation) "the slope" (co-efficient vector) should take a particular'
          'relatively high value, the λ term is forcing it to under-fit this "directionally slope" value')
    print('In conclusion, we indicate that for these values of λ, by decreasing the co-efficients of predictors '
          'aren\'t helping the model, but are limiting it\'s performance to fit well')

    print('\nNotes:\n'
          '* The Loss in the test set has an ascending trend while λ is increasing. That\'s due to over fitting '
          '  reduction and well generalization of our models\n'
          '* Our data isn\'t good enough and it\'s not well generalized. Plus our input features aren\'t '
          '  scattered well and don\'t represent a good feature extracted data!\n'
          '* There\'s an outlier values within the prices - The 535e3[$] rental price, which affects our regressors '
          '  badly! Highly affects our Loss values and hence, affects our Gradient -> learning rate -> weights update '
          '  -> the regressor vector etc.')
