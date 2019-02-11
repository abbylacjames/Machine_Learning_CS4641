from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from termcolor import colored, cprint
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np
from sklearn.preprocessing import StandardScaler



grid_param = {
    "activation":["logistic"],
    "max_iter":[200,250,300,350,375,400,425,450,500,550,600,700,800]
}
##======Uncomment to run Diabetes dataset==========================
# dataset = pd.read_csv("./csv_result-diabetes.csv")
# X = dataset.drop('class', axis=1)
# y = dataset['class']

##========Uncomment to run Wine Quality ================
dataset = pd.read_csv("./winequality-red.csv")
X = dataset.drop('quality', axis=1)
y = dataset['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


##================ GridSearch
classifier = MLPClassifier()
start_time = time.time()
gd_sr = GridSearchCV(estimator=classifier,
                     param_grid=grid_param,
                     scoring='accuracy',
                     cv=5,
                     n_jobs=-1)
gd_sr.fit(X_train, y_train)
best_parameters = gd_sr.best_params_
best_result = gd_sr.best_score_

# # ======Print out helpful data from the grid search===============
cprint("Training time: {0} \n".format(time.time() - start_time), "blue")
best_params = gd_sr.best_params_
best_result = gd_sr.best_score_
cprint("best parameters were: {0}".format(best_params), 'red')
cprint("best results were: {0}".format(best_result), 'red')

# ==============Make and run the classifier with the ideal parameters=================
max_iterations = best_params["max_iter"]
mlp_classifier = MLPClassifier(activation = 'logistic')
mlp_classifier =mlp_classifier.fit(X_train, y_train.values.ravel())
predicted = mlp_classifier.predict(X_test)
print(accuracy_score(y_test, predicted))
#####===================compare iterations - Diabetes===============
# iArray = []
# accuracy =[]
# for i in range(200,450):
#     mlp = MLPClassifier(activation='logistic',max_iter=i)
#     decision_tree = mlp.fit(X_train, y_train.values.ravel())
#     mlp = decision_tree.predict(X_test)
#     iArray.append(i)
#     accuracy.append(accuracy_score(y_test, predicted))
# plt.plot(iArray, accuracy, color='red', linestyle='solid', marker='x', markerfacecolor='red', markersize=5)
# title = "Diabetes" + ": Accuracy vs. Max Iterations " + 'NeuralNet'
# plt.title(title)
# plt.xlabel('Maximum Iterations')
# plt.ylabel('Accuracy')
# plt.show()

#####===================compare iterations - Wine Quality===============
# iArray = []
# accuracy =[]
# for i in range(200,450):
#     decision_tree = MLPClassifier(activation='logistic',max_iter=i)
#     decision_tree = decision_tree.fit(X_train, y_train.values.ravel())
#     predicted = decision_tree.predict(X_test)
#     iArray.append(i)
#     accuracy.append(accuracy_score(y_test, predicted))
# plt.plot(iArray, accuracy, color='red', linestyle='solid', marker='x', markerfacecolor='red', markersize=5)
# title = "Wine Quality" + ": Accuracy vs. Max Iterations " + 'NeuralNet'
# plt.title(title)
# plt.xlabel('Maximum Iterations')
# plt.ylabel('Accuracy')
# plt.show()

# #================== Create Learning Curve for Diabetes===================
# mlp = MLPClassifier(activation='logistic')
# i = 0.01
# training_accuracy_array = []
# testing_accuracy_array =  []
# iValues = []
# while i < 1:
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i, shuffle=True)
#     scaler = StandardScaler()
#     scaler.fit(X_train)
#     X_train = scaler.transform(X_train)
#     X_test = scaler.transform(X_test)
#     trained_model = mlp.fit(X_train, y_train.values.ravel())
#     predictions = trained_model.predict(X_test)
#     training_accuracy = 1 - accuracy_score(y_train,trained_model.predict(X_train))
#     testing_accuracy = 1 - accuracy_score(y_test,predictions)
#     training_accuracy_array.append (training_accuracy)
#     testing_accuracy_array.append(testing_accuracy)
#     iValues.append(1 - i)
#     i += 0.01
# plt.figure(figsize=(12, 6))
# plt.plot(iValues, training_accuracy_array, color='red', linestyle='solid', marker='x', markerfacecolor='red',
#              markersize=5)
# plt.plot(iValues, testing_accuracy_array, color='green', linestyle='solid', marker='x', markerfacecolor='green',
#              markersize=5)
# title = "Diabetes" + ": Training Error and Testing Error vs. Training size - " + 'Neural Network'
# plt.title(title)
# plt.xlabel('Training size (percentage of Data')
# plt.ylabel('Error Rate')
# plt.show()

#================== Create Learning Curve for Wine Quality===================

# mlp = MLPClassifier(activation='logistic')
# i = 0.01
# training_accuracy_array = []
# testing_accuracy_array =  []
# iValues = []
# while i < 1:
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i, shuffle=True)
#     scaler = StandardScaler()
#     scaler.fit(X_train)
#     X_train = scaler.transform(X_train)
#     X_test = scaler.transform(X_test)
#     trained_model = mlp.fit(X_train, y_train.values.ravel())
#     predictions = trained_model.predict(X_test)
#     training_accuracy = 1 - accuracy_score(y_train,trained_model.predict(X_train))
#     testing_accuracy = 1 - accuracy_score(y_test,predictions)
#     training_accuracy_array.append (training_accuracy)
#     testing_accuracy_array.append(testing_accuracy)
#     iValues.append(1 - i)
#     i += 0.01
# plt.figure(figsize=(12, 6))
# plt.plot(iValues, training_accuracy_array, color='red', linestyle='solid', marker='x', markerfacecolor='red',
#              markersize=5)
# plt.plot(iValues, testing_accuracy_array, color='green', linestyle='solid', marker='x', markerfacecolor='green',
#              markersize=5)
# title = "Wine Quality" + ": Training Error and Testing Error vs. Training size - " + 'SVM'
# plt.title(title)
# plt.xlabel('Training size (percentage of Data')
# plt.ylabel('Error Rate')
# plt.show()