import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn import neighbors
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score  # will be used to separate training and test
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn import preprocessing
from matplotlib.colors import ListedColormap


# load iris data
iris = load_iris()

# center input data
iris.data = preprocessing.scale(iris.data, with_std=False, with_mean=True)


# The following code shows the program training a decision tree and its results in prediction
clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=300, min_samples_leaf=5, class_weight={0:1,1:1,2:1})
clf = clf.fit(iris.data, iris.target)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)  # score will be the accuracy

print("Decision tree scores:")
print(scores)
print()


# The following code shows the training of k-nearest neighbors and its prediction results.
# Here we use a uniform weighting setting (weights='uniform'):
# any neighbors weights the same in the majority voting aggregation.
n_neighbors = 11
clf_knn = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf_knn = clf_knn.fit(iris.data, iris.target)
scores = cross_val_score(clf_knn, iris.data, iris.target, cv=5)  # score will be the accuracy

print("KNN classifier (uniform weight) scores:")
print(scores)


# shows the model predictions
print("Showing KNN predictions over the data:")
for i in range(len(iris.target)):
    instance = (iris.data[i, :]).reshape(1, -1)
    predicted = clf_knn.predict(instance)[0]

    if iris.target[i] == predicted:
        print(str(i)+" ok "+str(iris.target_names[iris.target[i]]))
    else:
        print(str(i)+" not ok, true class: "+str(iris.target_names[iris.target[i]])+"; predicted: "+ str(iris.target_names[predicted]))

print("-- end of data prediction --")


# In the following code we use a varying weighting setting (weights='distance'):
# any neighbors weights inversely with its distance to the test instance in the majority voting aggregation.
n_neighbors = 11
clf_knn2 = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
clf_knn2.fit(iris.data, iris.target)
scores2 = cross_val_score(clf_knn2, iris.data, iris.target, cv=5,scoring='accuracy')  # score will be the accuracy

print("KNN classifier (distance weight) accuracy score:")
print(scores2)

print("Showing KNN predictions over the data:")
for i in range(len(iris.target)):
    instance = (iris.data[i, :]).reshape(1, -1)
    predicted2 = clf_knn2.predict(instance)[0]
    if iris.target[i] == predicted2:
        print(str(i)+" ok "+str(iris.target_names[iris.target[i]]))
    else:
        print(str(i)+" not ok, true class: "+str(iris.target_names[iris.target[i]])+"; predicted: "+str(iris.target_names[predicted]))

print("-- end of data prediction --")

# Defining gaussian kernel...

# sigma squared
# with a value below 0.5, the performance of gaussian kernel begins to degrade
sigma_sq = 0.8 ** 2


# function kernel
def Knn_gaussian_kernel(weights):

    return np.exp(- (weights ** 2) / (2 * sigma_sq))


# get training and test samples
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

# graphic
fig = plt.figure()
fig.suptitle('Accuracy in k-nn with number of neighbors and types of weighting', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)
ax.set_xlabel('n. neighbors')
ax.set_ylabel('accuracy')

A = np.zeros((len(y_train), 3), dtype=np.float)  # 3 arrays for storing accuracies for each type of weighting
i = 0  # parameter in the control of the different diagram (=matrix A column index)

# it contains the best classifier for each type of weight
best_knn_classifiers = []

weight_types = ['uniform', 'distance', 'gaussian']

for weight_type in weight_types:

    print("\n-weighting:" + str(weight_type))
    best_accuracy = 0
    best_k = 1
    best_knn_clf = None

    for n_neighbors in np.arange(1, len(y_train) + 1):

        clf_knn2 = neighbors.KNeighborsClassifier(n_neighbors, weights=Knn_gaussian_kernel if weight_type == "gaussian" else weight_type)
        clf_knn2.fit(X_train, y_train)

        index = n_neighbors - 1  # computes the matrix row index
        A[index, i] = clf_knn2.score(X_test, y_test)

        if best_accuracy < A[index, i]:
            best_accuracy = A[index, i]
            best_k = n_neighbors
            best_knn_clf = clf_knn2

        # print("k neighbors=" + str(n_neighbors))
        # print("accuracy=" + str(clf_knn2.score(X_test, y_test)))

    # optimal K value
    print("best k=" + str(best_k))
    print("accuracy=" + str(best_accuracy))

    best_knn_classifiers.append(best_knn_clf)

    plt.plot(np.arange(1, len(y_train) + 1), A[:, i])
    i = i + 1

plt.legend(['uniform', 'distance', 'gaussian'], loc='lower left')
plt.show()


# 2D PLOTTING
# this function plots in a 2D space the predictions of the classifier for the iris data
def plot_decision_regions(X, y, y_predicted, classifier, plot_title, resolution=0.02):

    # train the classifier for the new input transformed by PCA
    classifier.fit(X, y)

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    # get predictions for the 2D space
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    # setup marker generator and color map
    markers = ('s', 'x', 'o')
    colors = ('red', 'blue', 'lightgreen')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # show boundaries of the classifier for the target output
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

    # set x and y limits
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for index_actual, class_actual in enumerate(np.unique(y)):
        for index_predicted, class_predicted in enumerate(np.unique(y)):

            # get indexes of the target data belonging to the actual class to consider
            indexes_y_actual = [index for index, value in enumerate(y) if value == class_actual]

            # get indexes of the target data belonging to the predicted class to consider
            indexes_y_predicted = [index for index, value in enumerate(y_predicted) if value == class_predicted]

            # get the intersection of the indexes
            indexes_intersection = [value for value in indexes_y_predicted if value in indexes_y_actual]

            # skip plotting if there are no items
            if len(indexes_intersection) <= 0:
                continue

            # get points of intersection data
            ax_x = X[indexes_intersection, 0]
            ax_y = X[indexes_intersection, 1]

            # color of predicted class
            c = [cmap(index_predicted)]

            # marker for the actual class
            marker = markers[index_actual]

            # plot the points indicating the actual and predicted class
            plt.scatter(x=ax_x,
                        y=ax_y,
                        alpha=0.6,
                        c=c,
                        edgecolor='black',
                        marker=marker,
                        label="Actual: " + str(class_actual) + ", predicted: " + str(class_predicted))

    # show the figure
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='upper left')
    plt.title(plot_title)
    plt.show()


# initialize pca model
pca = PCA(n_components=2)

# used for 2D instances plotting
X_test_pca = pca.fit_transform(X_test)

# plot in 2D the
for index, best_knn_clf in enumerate(best_knn_classifiers):

    # get prediction for the classifier
    y_predicted = best_knn_clf.predict(X_test)

    plot_decision_regions(X_test_pca, y_test, y_predicted, best_knn_clf, weight_types[index])
