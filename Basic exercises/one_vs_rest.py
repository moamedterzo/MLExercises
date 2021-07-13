from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier as OvR
from sklearn.svm import LinearSVC
import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import clone


# custom OneVsRest classifier
class OneVsRestClassifier:

    classifiers = []

    def __init__(self, learner):
        self.learner = learner

    def fit(self, data, labels):

        # for each label value...
        for label_value in set(labels):

            # set labels for the OneVsRest classifier
            new_labels = [1 if label_value == label else -1 for label in labels]

            # train the classifier for this particular label value
            new_classifier = clone(self.learner)
            new_classifier.fit(data, new_labels)

            # append the classifier and the label value
            self.classifiers.append([new_classifier, label_value])

        return self

    def predict(self, data):

        # store in the matrix all classifiers' predictions
        matrix = []

        for classifier in self.classifiers:
            matrix.append(classifier[0].predict(data))

        # the matrix has for each row an instance, and each column corresponds to a classifier
        matrix = np.array(matrix).transpose()

        # for each instance, the predicted class is taken considering the classifier who returns the highest label value
        # In this case the highest label value is: 1
        result = []

        for instance_prediction in matrix:

            # if different classifiers predict the label '1', the first one is chosen
            classifier_index = np.argmax(instance_prediction)

            # append the original label value
            result.append(self.classifiers[classifier_index][1])

        return result


# load dataset
digits = datasets.load_digits()

# plot first 10 figures
for index, image in enumerate(digits.images[:10]):
    plt.subplot(2, 5, index + 1)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')

plt.show()


# create training and test set
X, y = digits.data[0:1000], digits.target[0:1000]
X_test, y_test = digits.data[1000:], digits.target[1000:]

# create OneVsRest classifier and print accuracy
binaryLearner = LinearSVC(random_state=0)

oneVrestLearningAlgorithm = OvR(binaryLearner)
oneVrestLearningAlgorithm.fit(X, y)
predicted_labels = oneVrestLearningAlgorithm.predict(X_test)

# n.b.: the above is equivalent to:
# predicted_labels = OvR(LinearSVC(random_state=0)).fit(X,y).predict(X_test)

print("Ovr accuracy: %s" % (1.0 - (np.count_nonzero(y_test - predicted_labels) / float(len(predicted_labels)))))


# EXERCISE

# creating custom OneVsRest classifier
ovr = OneVsRestClassifier(LinearSVC(random_state=0))

# training and prediction...
predicted_labels = ovr.fit(X, y).predict(X_test)

# accuracy result
print("OneVsRestClassifier accuracy: %s" % (1.0 - np.count_nonzero(predicted_labels - y_test) / float(len(y_test))))


