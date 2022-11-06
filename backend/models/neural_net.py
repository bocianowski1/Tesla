import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def neural_net_on(datasets, show_graph=True, show_training=False):
    for dataset in datasets:
        X = np.array(dataset[['KMS', 'Age']])
        y = np.array(dataset['Price'])
        y = y.reshape(y.shape[0], 1)

        n_samples, n_features = X.shape

        # splitting data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

        # scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = torch.from_numpy(X_train.astype(np.float32))
        X_test = torch.from_numpy(X_test.astype(np.float32))
        y_train = torch.from_numpy(y_train.astype(np.float32))
        y_test = torch.from_numpy(y_test.astype(np.float32))

        model = nn.Linear(n_features, 1)

        loss = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

        # epic training montage
        for epoch in range(250):
            y_pred = model(X_train)
            l = loss(y_pred, y_train)

            l.backward()

            optimizer.step()
            optimizer.zero_grad()
            
            if show_training and epoch % 100 == 0:
                print(f'\tepoch: {epoch}, loss: {l.item()}')


        prediction = model(X_test).detach().numpy()

        result_dict = {int(prediction[i]): int(y_test[i].numpy()) for i in range(len(prediction))}
        
        print(f'\n{dataset.name}')
        # print(f'  Accuracy: {accuracy(model(X_test), y_test)}')
        print(f'  4 first predictions: {list(result_dict.items())[:4]}\n')

        if show_graph:
            fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
            fig.suptitle(f'Neural Network on {dataset.name}')
            ax[0].plot(X_test, y_test, '.', color='#99c3a6')
            ax[0].set_title('Original Test Data')

            ax[1].plot(X_test, y_test, '.', color='#99c3a6')
            ax[1].plot(X_test, prediction, 'o', color='orange')
            ax[1].set_title('Prediction on Test Data')
            plt.show()