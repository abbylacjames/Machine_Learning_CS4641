# File to run k-nearest-neighbor on a dataset

import pandas as pd
import time
from termcolor import cprint
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import warnings
import matplotlib.pyplot as plt
import pylab

def main():

    # ==================================================== Dota Dataset =============================================================
    # dataset = pd.read_csv("./csv_result-diabetes.csv")
    # X = dataset.drop('class', axis=1)
    # y = dataset['class']
    dataset = pd.read_csv("./winequality-red.csv")
    X = dataset.drop('quality', axis=1)
    y = dataset['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # Scale data so it is uniformly evaluated
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Set up grid for finding the best hyperparameters with this classifier
    grid_param = {
        'n_neighbors': [10,15,20,25,30,35],
        'weights':['uniform','distance']
    }
    knn = KNeighborsClassifier()
    start_time = time.time()
    gd_sr = GridSearchCV(estimator=knn, param_grid=grid_param, scoring='accuracy', cv=3, n_jobs=-1)
    gd_sr.fit(X_train, y_train.values.ravel())

    # Print out helpful data from the grid search
    print "===== Dota Dataset ====="
    cprint("Training time: {0} \n".format(time.time() - start_time), "blue")
    best_params = gd_sr.best_params_
    best_result = gd_sr.best_score_
    cprint("best parameters were: {0}".format(best_params), 'red')
    cprint("best results were: {0}".format(best_result), 'red')

    # Make and run the classifier with the ideal parameters
    # Use the DTC ideal parameters from the decision tree
    neighbors = best_params['n_neighbors']
    P = best_params['p']
    knn = KNeighborsClassifier(n_neighbors=neighbors, p=P)
    knn.fit(X_train, y_train.values.ravel())



    #============= showing accuracy based on k ===============
    # err = []
    #
    # # Calculating error for K values between 1 and 40
    # for i in range(1, 40):
    #     knn = KNeighborsClassifier(n_neighbors=i)
    #     knn.fit(X_train, y_train)
    #     pred_i = knn.predict(X_test)
    #     err.append(accuracy_score(y_test, pred_i))
    #
    # plt.figure(figsize=(12, 6))
    # plt.plot(range(1, 40), err, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
    # plt.title('Error Rate K Value')
    # plt.xlabel('K Value')
    # plt.ylabel('Mean Error')
    # pylab.show()
    #======================================================




if __name__ == "__main__":
    main()