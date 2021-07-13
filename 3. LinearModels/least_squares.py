from sklearn import datasets
from scipy.optimize import fmin_bfgs
import numpy as np
from numpy.linalg import norm
from numpy.linalg import inv


# load Boston dataset
boston = datasets.load_boston()
data = np.array(boston.data)

# print description
# print(boston.DESCR)

# append a column of 1 to the variable 'data'
t = np.ones(len(data)).reshape(len(data), 1)
data = np.append(data, t, 1)

# target values
target = np.array(boston.target)

# We now divide the data into a training set and a test set
X, y = data[0:400, :], target[0:400]
X_test, y_test = data[400:, :], target[400:]


# store in a variable the result of: X' * X
Xtranspose_dot_X = np.transpose(X).dot(X)

# (ex. 1) calculate the weights using the least squares regression formula
w_least = inv(Xtranspose_dot_X).dot(np.transpose(X)).dot(y)

# lambda parameter for the ridge regression
lambda_par = 0.01

# (ex. 2) calculate the weights using the ridge regression formula
w_ridge = inv(Xtranspose_dot_X + lambda_par * np.identity(len(Xtranspose_dot_X))).dot(np.transpose(X)).dot(y)


# (ex. 3) define the function to be minimized
def f_lasso(w):
    return np.sum((X.dot(w) - y) ** 2) + norm(w, ord=1)


# calculate the weights using the lasso regression minimization algorithm
w_lasso = fmin_bfgs(f_lasso, np.zeros(14))

# print the calculated weights
# print("Least: " + str(w_least))
# print("Ridge: " + str(w_ridge))
# print("Lasso: " + str(w_lasso))


# calculating statistics for training and test set

def compute_S_statistic(y_actual, y_predicted):

    # compute the S statistic, defined by the exercise
    return (np.sum((y_actual - y_predicted) ** 2) / len(y_actual)) ** (1/2)


stat_least_training = compute_S_statistic(y, X.dot(w_least))
stat_least_test = compute_S_statistic(y_test, X_test.dot(w_least))

stat_ridge_training = compute_S_statistic(y, X.dot(w_ridge))
stat_ridge_test = compute_S_statistic(y_test, X_test.dot(w_ridge))

stat_lasso_training = compute_S_statistic(y, X.dot(w_lasso))
stat_lasso_test = compute_S_statistic(y_test, X_test.dot(w_lasso))


# print the calculated statistics
print()

print("Least squares statistic on training set:    " + str(stat_least_training))
print("Least squares statistic on test set:        " + str(stat_least_test), '\n')

print("Ridge regression statistic on training set: " + str(stat_ridge_training))
print("Ridge regression statistic on test set:     " + str(stat_ridge_test), '\n')

print("Lasso regression statistic on training set: " + str(stat_lasso_training))
print("Lasso regression statistic on test set:     " + str(stat_lasso_test), '\n')
