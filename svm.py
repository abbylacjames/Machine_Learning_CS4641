import pandas as pd
import numpy as np
import time
from termcolor import cprint
from sklearn import svm
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
import matplotlib.pyplot as plt
import pylab
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np

def main():
    #=======Uncomment to run with Wine=========
    # dataset = pd.read_csv("./winequality-red.csv")
    # X = dataset.drop('quality', axis=1)
    # y = dataset['quality']

    #=======Uncomment to run with diabetes
    dataset = pd.read_csv("./csv_result-diabetes.csv")
    X = dataset.drop('class', axis=1)
    y = dataset['class']

    #============================

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)

    # Scale data so it is uniformly evaluated
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Set up grid for finding the best hyperparameters with this classifier
    # grid_param = {
    #     'kernel': ['rbf'],
    #     'gamma':[0.001,0.1,0.25,0.3,0.75,1],
    #     'C':[1,3,5,10,50,100,1000]
    # }
    # classifier = svm.SVC()
    # start_time = time.time()
    # gd_sr = GridSearchCV(estimator=classifier, param_grid=grid_param, scoring='accuracy', cv=3, n_jobs=-1)
    # gd_sr.fit(X_train, y_train.values.ravel())
    #
    # # Print out helpful data from the grid search
    # cprint("Training time: {0} \n".format(time.time() - start_time), "blue")
    # best_params = gd_sr.best_params_
    # best_result = gd_sr.best_score_
    # cprint("best parameters were: {0}".format(best_params), 'red')
    # cprint("best results were: {0}".format(best_result), 'red')

    # Make and run the classifier with the ideal parameters
    # Use the DTC ideal parameters from the decision tree
    # kern = best_params['kernel']
    # c = best_params['C']
    # gamma = best_params['gamma']
    # classifier = svm.SVC(kernel=kern, gamma = gamma, C =c)
    # classifier.fit(X_train, y_train.values.ravel())
    # predicted = classifier.predict(X_test)
    # print(accuracy_score(y_test, predicted))
    # classifier = svm.SVC()
    # classifier.fit(X_train, y_train.values.ravel())
    # predicted = classifier.predict(X_test)
    # print(accuracy_score(y_test, predicted))


    ## ========Learning Curve Diabetes =========
    # classifier = svm.SVC(kernel='rbf', gamma=0.75, C=1)
    # i = 0.01
    # training_accuracy_array = []
    # testing_accuracy_array = []
    # iValues = []
    # while i < 1:
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i, shuffle=True)
    #     scaler = StandardScaler()
    #     scaler.fit(X_train)
    #     X_train = scaler.transform(X_train)
    #     X_test = scaler.transform(X_test)
    #     trained_model = classifier.fit(X_train, y_train.values.ravel())
    #     predictions = trained_model.predict(X_test)
    #     training_accuracy = 1 - accuracy_score(y_train, trained_model.predict(X_train))
    #     testing_accuracy = 1 - accuracy_score(y_test, predictions)
    #     training_accuracy_array.append(training_accuracy)
    #     testing_accuracy_array.append(testing_accuracy)
    #     iValues.append(1 - i)
    #     i += 0.01
    #
    # plt.figure(figsize=(12, 6))
    # plt.plot(iValues, training_accuracy_array, color='red', linestyle='solid', marker='x', markerfacecolor='red',
    #          markersize=5)
    # plt.plot(iValues, testing_accuracy_array, color='green', linestyle='solid', marker='x', markerfacecolor='green',
    #          markersize=5)
    # title = "Diabetes" + ": Training Error and Testing Error vs. Training size - " + 'SVM'
    # plt.title(title)
    # plt.xlabel('Training size (percentage of Data')
    # plt.ylabel('Error Rate')
    # plt.show()

    #=========== Learning curve Wine ==================
    classifier = svm.SVC(kernel='rbf', gamma=0.75, C=1)
    i = 0.01
    training_accuracy_array = []
    testing_accuracy_array =  []
    iValues = []
    while i < 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i, shuffle=True)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        trained_model = classifier.fit(X_train, y_train.values.ravel())
        predictions = trained_model.predict(X_test)
        training_accuracy = 1 - accuracy_score(y_train,trained_model.predict(X_train))
        testing_accuracy = 1 - accuracy_score(y_test,predictions)
        training_accuracy_array.append (training_accuracy)
        testing_accuracy_array.append(testing_accuracy)
        iValues.append(1 - i)
        i += 0.01
    plt.figure(figsize=(12, 6))
    plt.plot(iValues, training_accuracy_array, color='red', linestyle='solid', marker='x', markerfacecolor='red',
             markersize=5)
    plt.plot(iValues, testing_accuracy_array, color='green', linestyle='solid', marker='x', markerfacecolor='green',
             markersize=5)
    title = "Wine Quality" + ": Training Error and Testing Error vs. Training size - " + 'SVM'
    plt.title(title)
    plt.xlabel('Training size (percentage of Data')
    plt.ylabel('Error Rate')
    plt.show()

if __name__ == "__main__":
    main()