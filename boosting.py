# File to run boosting on a dataset

import pandas as pd
import time
from termcolor import colored, cprint
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import pylab

def main():

    # ==================================================== Run Diabetes Set =============================================================
    # dataset = pd.read_csv("./csv_result-diabetes.csv")
    # X = dataset.drop('class', axis=1)
    # y = dataset['class']

    # ==================================================== Run Diabetes Set =============================================================
    dataset = pd.read_csv("./winequality-red.csv")
    X = dataset.drop('quality', axis=1)
    y = dataset['quality']

    #========================================================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


    # Scale data so it is uniformly evaluated
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    #==============Run GridSearch for Cross Validation =======================
    # Set up grid for finding the best hyperparameters with this classifier
    grid_param = {
        'learning_rate':[1,2,3,4,5,6,7,8,9,10],
        "n_estimators": [20,30,40, 45, 50, 55, 60, 65, 70, 75, 80]
    }

    # dtc = DecisionTreeClassifier()
    # booster = AdaBoostClassifier(base_estimator=dtc)
    # start_time = time.time()
    # gd_sr = GridSearchCV(estimator=booster, param_grid=grid_param, scoring='accuracy', cv=3, n_jobs=-1)
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
    # dtc = DecisionTreeClassifier()
    # estimators = best_params["n_estimators"]
    # rates = best_params["learning_rate"]
    # booster = AdaBoostClassifier(base_estimator=dtc, n_estimators=estimators,learning_rate=rates)
    # booster.fit(X_train, y_train.values.ravel())
    # predicted = booster.predict(X_test)
    # print(accuracy_score(y_test, predicted))
    # booster.fit(X_train, y_train.values.ravel())
    # predicted = booster.predict(X_test)
    # print(accuracy_score(y_test, predicted))

    #Create Graph comparing hyperparameters to accuracy ======================
    iArray = []
    accuracy = []
    for i in range(1, 100):
        decision_tree = DecisionTreeClassifier()
        booster = AdaBoostClassifier(base_estimator=decision_tree, n_estimators=i)
        booster = decision_tree.fit(X_train, y_train.values.ravel())
        predicted = booster.predict(X_test)
        iArray.append(i)
        accuracy.append(accuracy_score(y_test, predicted))
        print ("i")
        print (i)
        print ("accuracy")
        print(accuracy_score(y_test, predicted))
    plt.plot(iArray, accuracy, color='red', linestyle='solid', marker='x', markerfacecolor='red', markersize=5)
    title = "Wine Quality" + ": Accuracy vs. Learning Rate- " + 'Boosting'
    plt.title(title)
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.show()





if __name__ == '__main__':
    main()