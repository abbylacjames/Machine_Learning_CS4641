import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from termcolor import colored, cprint
import time
from sklearn.model_selection import learning_curve

#
# 'max_leaf_nodes':[2, 3, 4, 5, 6, 7, 8, 9, 10],
#'min_samples_leaf':[50,52,55,57,58,60]
#max-depth 7 and max_leaf_nodes 30
#'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#max-depth 5
#max_depth 5, min_samples lead 30

grid_param = {
    'max_depth':[350,360,370,380,390,400,410,420],
    'max_leaf_nodes':[300,320,340,360,380,400,420,440,460]
}
#==== Uncomment to Run Diabetes ======
# dataset = pd.read_csv("./csv_result-diabetes.csv")
# X = dataset.drop('class', axis=1)
# y = dataset['class']

#====Uncomment to Run Wine =========
dataset = pd.read_csv("./winequality-red.csv")
X = dataset.drop('quality', axis=1)
y = dataset['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Scale data so it is uniformly evaluated
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
decision_tree = DecisionTreeClassifier()
start_time = time.time()
gd_sr = GridSearchCV(estimator=decision_tree, param_grid=grid_param, scoring='accuracy', cv=3, n_jobs=-1)
gd_sr.fit(X_train, y_train.values.ravel())

# Print out helpful data from the grid search
cprint("Training time: {0} \n".format(time.time() - start_time), "blue")
best_params = gd_sr.best_params_
best_result = gd_sr.best_score_
cprint("best parameters were: {0}".format(best_params), 'red')
cprint("best results were: {0}".format(best_result), 'red')

# Make and run the classifier with the ideal parameters
depth = best_params["max_depth"]
crit = best_params['max_leaf_nodes']
decision_tree = DecisionTreeClassifier(max_leaf_nodes=crit,max_depth=depth)
decision_tree =decision_tree.fit(X_train, y_train.values.ravel())
predicted = decision_tree.predict(X_test)
print(accuracy_score(y_test, predicted))
decision_tree = DecisionTreeClassifier()
decision_tree = decision_tree.fit(X_train, y_train.values.ravel())
predicted = decision_tree.predict(X_test)
print(accuracy_score(y_test, predicted))
iArray = []
accuracy = []
for i in range(2,500):
    decision_tree = DecisionTreeClassifier(max_depth = i)
    decision_tree = decision_tree.fit(X_train, y_train.values.ravel())
    predicted = decision_tree.predict(X_test)
    iArray.append(i)
    accuracy.append(accuracy_score(y_test, predicted))
    print ("i")
    print (i)
    print ("accuracy")
    print(accuracy_score(y_test, predicted))
plt.plot(iArray, accuracy, color='red', linestyle='solid', marker='x', markerfacecolor='red', markersize=5)
title = "Wine Quality" + ": Accuracy vs. Maximum Depth- " + 'Decision Tree'
plt.title(title)
plt.xlabel('Maximum Depth')
plt.ylabel('Accuracy')
plt.show()


# #Curve
# train_sizes, train_scores, test_scores = learning_curve(decision_tree, X, y, train_sizes=np.linspace(.05, .95, num=19), n_jobs=None)
# train_scores_mean = np.mean(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# plt.figure(figsize=(12, 6))
# plt.plot(train_sizes, 1- train_scores_mean, color='red', linestyle='solid', marker='x', markerfacecolor='red', markersize=5)
# plt.plot(train_sizes, 1- test_scores_mean, color='green', linestyle='solid', marker='x', markerfacecolor='green', markersize=5)
# title = "Wine Quality" + ": Training Error and Testing Error vs. Training size - " + 'Decision Tree'
# plt.title(title)
# plt.xlabel('Training size')
# plt.ylabel('Error Rate')
# plt.show()