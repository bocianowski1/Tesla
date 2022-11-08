import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression

def linear_regression_on(datasets, show_graph=True):
    for dataset in datasets:
        X = dataset.drop(['Price'], axis=1)
        y = dataset['Price']

        # splitting data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

        # scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # defining linear regressor and fitting
        linear_regressor = LinearRegression().fit(X_train, y_train)
        prediction = linear_regressor.predict(X_test)
        result_dict = {int(prediction[i]): list(y_test)[i] for i in range(len(prediction))}
        
        print(f'{dataset.name}')
        print(f'  Score: {round(linear_regressor.score(X_test, y_test), 4)}')
        print(f'  4 first predictions: {list(result_dict.items())[:4]}\n')
        
        a, b = linear_regressor.coef_, linear_regressor.intercept_
        if show_graph:
            fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
            fig.suptitle(f'Multivariate Regression on {dataset.name}')
            ax[0].plot(X_test, y_test, '.', color='#99c3a6')
            ax[0].set_title('Original Test Data')

            ax[1].plot(X_test, y_test, '.', color='#99c3a6')
            ax[1].plot(X_test, prediction, 'o', color='orange')
            ax[1].set_title('Prediction on Test Data')
            plt.show()