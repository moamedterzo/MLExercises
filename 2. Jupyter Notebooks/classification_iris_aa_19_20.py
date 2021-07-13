from sklearn import tree
from sklearn.datasets import load_iris
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score  # will be used to separate training and test
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import graphviz


# load iris dataset
iris = load_iris()

# Generate a random permutation of the indices of examples that will be later used
# for the training and the test set
np.random.seed(0)
indices = np.random.permutation(len(iris.data))


# We now decide to keep the last 10 indices for test set, the remaining for the training set
NUMBER_OF_TEST_SAMPLES = 30

indices_training = indices[:-NUMBER_OF_TEST_SAMPLES]
indices_test = indices[-NUMBER_OF_TEST_SAMPLES:]

iris_X_train = iris.data[indices_training]  # keep for training all the matrix elements with the exception of the last 10
iris_y_train = iris.target[indices_training]
iris_X_test = iris.data[indices_test]  # keep the last 10 elements for test set
iris_y_test = iris.target[indices_test]


def print_classifier_performance(clf, graphviz_file_suffix):

    # apply fitted model "clf" to the test set
    predicted_y_test = clf.predict(iris_X_test)

    # print the predictions (class numbers associated to classes names in target names)
    print("Predictions:")
    print(predicted_y_test)
    print("True classes:")
    print(iris_y_test)

    # print(iris.target_names)

    # print the corresponding instances indexes and class names
    # for i in range(len(iris_y_test)):
    #    print("Instance # " + str(indices_test[i]) + ": ")
    #    print("Predicted: " + iris.target_names[predicted_y_test[i]] + "\t True: " + iris.target_names[iris_y_test[i]] + "\n")


    # Look at the specific examples
    # for i in range(len(iris_y_test)):
    #     print("Instance # " + str(indices_test) + ": ")
    #     s = ""
    #     for j in range(len(iris.feature_names)):
    #         s = s + iris.feature_names[j] + "=" + str(iris_X_test[i][j])
    #         if (j < len(iris.feature_names) - 1): s = s + ", "
    #     print(s)
    #     print("Predicted: " + iris.target_names[predicted_y_test[i]] + "\t True: " + iris.target_names[
    #         iris_y_test[i]] + "\n")

    print("Scores...")

    # print some metrics results
    acc_score = accuracy_score(iris_y_test, predicted_y_test)
    print("Accuracy score: " + str(acc_score))

    f1 = f1_score(iris_y_test, predicted_y_test, average='macro')
    print("F1 score: " + str(f1))

    # use cross-validation
    scores = cross_val_score(clf, iris.data, iris.target, cv=5)  # score will be the accuracy
    print("Cross-validation accuracy scores: " + str(scores))

    # computes F1- score
    f1_scores = cross_val_score(clf, iris.data, iris.target, cv=5, scoring='f1_macro')
    print("Cross validation F1 scores: " + str(f1_scores))

    # (ex 4) print the confusion matrix
    print("Confusion matrix")
    print("the class order for the columns and the rows is:", list(iris.target_names), '\n')
    print(confusion_matrix(iris_y_test, predicted_y_test))


    # == Print tree in a PDF file ==

    # dot_data = tree.export_graphviz(clf, out_file=None)
    # graph = graphviz.Source(dot_data)
    # graph.render("my_iris_predictions")

    # print(list(iris.feature_names))

    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=iris.feature_names,
                                    class_names=iris.target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("my_iris_predictions_" + graphviz_file_suffix)

    # ROC curve (ex 5)

    # for each class a curve is plotted
    colors = ('red', 'blue', 'lightgreen')

    for class_index in clf.classes_:

        # contains, for each node, the number of positives and negatives for a given class
        probability_nodes = []

        # for each node...
        for i in range(clf.tree_.node_count):

            # if it is a leaf...
            if clf.tree_.children_left[i] == -1:

                number_of_positives = clf.tree_.value[i][0][class_index]
                total_of_elements = sum(clf.tree_.value[i][0])

                probability_nodes.append([number_of_positives, total_of_elements - number_of_positives])

        # the tree leafs are ordered based on the the class probability estimation
        probability_nodes = sort_tree_nodes_probabilities(np.array(probability_nodes))

        # get ROC points
        false_positive_rate_points, true_positive_rate_points = create_roc_points(probability_nodes)

        # plot the ROC points for the given class
        plt.plot(false_positive_rate_points, true_positive_rate_points,
                    c=colors[class_index], label=iris.target_names[class_index],
                    marker='o', alpha=0.7)

        plt.legend(iris.target_names[class_index])

    plt.legend()
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.show()

    return clf


def create_roc_points(probability_nodes):

    # used for the ratio
    total_of_positives = float(sum(probability_nodes[:, 0]))
    total_of_negatives = float(sum(probability_nodes[:, 1]))

    x_points = []
    y_points = []

    # each point represents the False positive rate against the True positive rate, considering all positive the instances of nodes
    # before or at position i, and all negative the instances of nodes after position i
    for i in range(len(probability_nodes) + 1):

        x_points.append(sum(probability_nodes[:i, 1]))
        y_points.append(sum(probability_nodes[:i, 0]))

    # the ratio is multiplied by 100 in order to prevent to round small numbers to zero
    return [np.array(x_points) * 100 / total_of_negatives, np.array(y_points) * 100 / total_of_positives]


def sort_tree_nodes_probabilities(node_probabilities):

    predicate = node_probabilities[:, 1] / (node_probabilities[:, 0] + 1)  # negatives divided by positives
    order = np.argsort(predicate)

    return node_probabilities[order]


def get_inflated_training_set(X_train, y_train, inflate_weight):

    # the inflated classes are 1 and 2 (versicolor and virginica)
    instances_to_inflate = [True if y == 1 or y == 2 else False for y in y_train]
    X_train_to_inflate = X_train[instances_to_inflate]
    y_train_to_inflate = y_train[instances_to_inflate]

    new_X_train = np.copy(X_train)
    new_y_train = np.copy(y_train)

    i = 0
    while i < inflate_weight:
        i += 1

        new_X_train = np.concatenate((new_X_train, X_train_to_inflate))
        new_y_train = np.concatenate((new_y_train, y_train_to_inflate))

    return [new_X_train, new_y_train]


# presentation classifier
print(" == CLASSIFIER 1 (standard parameters) ==\n")
clf1 = tree.DecisionTreeClassifier(criterion="entropy", random_state=300, min_samples_leaf=5, class_weight={0:1,1:1,2:1})

clf1.fit(iris_X_train, iris_y_train)
print_classifier_performance(clf1, 'clf1')


# ex 1 e 4
print(" == CLASSIFIER 2 (inflated training set) ==\n")
clf2 = tree.DecisionTreeClassifier(criterion="entropy", random_state=400, min_samples_leaf=5,max_depth=4, max_leaf_nodes = 5, class_weight={0:1,1:1,2:1})

# get inflated training set...
clf2_iris_X_train, clf2_iris_y_train = get_inflated_training_set(iris_X_train, iris_y_train, 10)

clf2.fit(clf2_iris_X_train, clf2_iris_y_train)
print_classifier_performance(clf2, 'clf2')


# ex 2
print(" == CLASSIFIER 3 (different class weights) ==\n")
clf3 = tree.DecisionTreeClassifier(criterion="entropy", random_state=500, min_samples_leaf=2, class_weight={0:1,1:10,2:10})

clf3.fit(iris_X_train, iris_y_train)
print_classifier_performance(clf3, 'clf3')









