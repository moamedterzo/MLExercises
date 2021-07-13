import numpy as np
import math
import copy
import sys
import sklearn.datasets
from sklearn.svm import SVC
import random
from sklearn import preprocessing

np.set_printoptions(threshold=sys.maxsize)


class SVC_custom:
    def __init__(self, kernel="rbf", degree="3", coef0=0):
        self.svc = SVC(kernel=kernel, degree=degree, coef0=coef0, gamma="scale")

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
           sample_weight = sample_weight * len(X)

        self.svc.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.svc.predict(X)


class AdaBoost:
    def __init__(self, weakModel, T):

        self.weakModel = weakModel
        self.T = T
        self.models = []

    def fit(self, X, y, K_print = None):

        # weights vector
        sample_weight = np.ones(len(X)) / len(X)

        # model counter
        t = 0

        while t < self.T:

            # print every K iterations
            flag_print = K_print is not None and t % K_print == 0

            if flag_print:
                print("Training " + str(t) + "-model")

            # deep copy the model
            model = copy.deepcopy(self.weakModel)

            # train the model with the specified weights
            model.fit(X, y, sample_weight=sample_weight)

            # get predicted values
            y_pred = model.predict(X)

            # weighted error
            err = sum(sample_weight * [1 if yy_pred != y[idx] else 0 for idx, yy_pred in enumerate(y_pred)])

            # model reliability
            alfa = np.log((1 - err) / err) / 2

            # example re-weighting
            for i in range(len(sample_weight)):
                sample_weight[i] = sample_weight[i] * np.exp(-alfa * y_pred[i] * y[i])

            # weights normalization
            sample_weight = sample_weight / sum(sample_weight)

            # add model to the set of models (along with alfa value)
            self.models.append([alfa, model])

            t+= 1

            # print currect score every K iterations
            if flag_print:
                print("Current loss error is: " + str(err))
                print("Curent model alfa is: " + str(alfa))

                # current AdaBoost score
                print("Current AdaBoost score is:")
                y_pred = self.predict(X)
                self.print_scores(y, y_pred)
                print()

        return self

    def predict(self, X):

        # vector containing the sum of the models predictions
        y_pred = np.zeros(len(X))

        for i in range(len(self.models)):

            model = self.models[i][1]
            alfa = self.models[i][0]

            # add to the final prediction, the weighted prediction of current model, weighted by its reliability alfa
            y_pred = np.add(y_pred, model.predict(X) * alfa)

        # step function, in order to return for every instance 1 or -1
        return [1 if y >= 0 else -1 for y in y_pred]

    def print_scores(self, y_actual, y_predicted):

        n_samples = len(y_predicted)
        errors = (1 - np.dot(y_actual, y_predicted) / n_samples) / 2

        print("% of errors: " + str(errors * 100))


# get training and test set
X, y = sklearn.datasets.make_hastie_10_2()

X_train = X[0:8000, :]
y_train = y[0:8000]
X_test = X[8000:, :]
y_test = np.array(y[8000:])


def ex1():

    # create weak model
    weakModel = SVC_custom(kernel="poly", degree=3, coef0=0)

    # create AdaBoost model using the weak model
    adaboost = AdaBoost(weakModel, 100)
    adaboost.fit(X_train, y_train, K_print=10)

    # predict values
    predicted_y_train = adaboost.predict(X_train)
    predicted_y_test = adaboost.predict(X_test)

    # print scores
    print("SCORES ON TRAINING SET")
    adaboost.print_scores(y_train, predicted_y_train)

    print("SCORES ON TEST SET")
    adaboost.print_scores(y_test, predicted_y_test)


class RandomLinearModel:

    def __init__(self):
        self.w = []
        self.t = 0

    # returns the weighted loss
    def loss(self, y, y_pred, sample_weight):

        return sum(sample_weight * [1 if yy_pred != y[idx] else 0 for idx, yy_pred in enumerate(y_pred)])

    def fit(self, X, y, sample_weight):

        # get random values for w and t, sampling from U(-1, 1)
        number_of_features = np.shape(X)[1]

        for i in range(number_of_features):
            self.w.append(random.uniform(-1, 1))

        self.w = np.array(self.w)
        self.t = random.uniform(-1, 1)

        # predict values after having initialized the parameters
        y_pred = self.predict(X)

        # flip the model's parameters if the loss is big enough
        if self.loss(y, y_pred, sample_weight) > 0.5:
            self.w = self.w * -1
            self.t = self.t * -1

        return self

    def predict(self, X):

        # returns -1 or 1 based on the sign of the classification
        # this is a linear classification, with a step activation function
        return np.array([-1 if y < 0 else 1 for y in (np.dot(X, self.w) + self.t)])


def ex2():
    # Training the random linear model
    rs = RandomLinearModel()
    a = AdaBoost(rs, 10000)
    a.fit(X_train, y_train, 2000)

    # printing scores
    y_train_pred = a.predict(X_train)
    y_test_pred = a.predict(X_test)

    print("SCORES ON TRAINING SET")
    a.print_scores(y_train, y_train_pred)

    print("SCORES ON TEST SET")
    a.print_scores(y_test, y_test_pred)

ex1()
ex2()

