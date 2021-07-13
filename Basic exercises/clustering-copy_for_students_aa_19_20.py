import csv
from os.path import join

import numpy as np

from sklearn.cluster import KMeans, SpectralClustering, MeanShift
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.random import sample_without_replacement


# this function reads the data file, loads the configuration attributes specifiefd in the heading
# (numer of examples and features), the list of feature names
# and loads the data in a matrix named data
def load_data(file_path, file_name):
    with open(join(file_path, file_name)) as csv_file:
        data_file = csv.reader(csv_file, delimiter=',')
        temp1 = next(data_file)
        n_samples = int(temp1[0])
        print("n_samples=")
        print(n_samples)
        n_features = int(temp1[1])
        temp2 = next(data_file)
        feature_names = np.array(temp2[:n_features])

        data_list = [iter for iter in data_file]

        data = np.asarray(data_list, dtype=np.float64)

    return data, feature_names, n_samples, n_features


# The main program reads the input file containing the dataset
# file_path is the file path where the file with the data to be read are located
# we assume the file contains an example per line
# each example is a list of real values separated by a comma (csv format)
# The first line of the file contains the heading with:
# N_samples,n_features,
# The second line contains the feature names separated by a comma

# file_path="~/meo/Documents/Didattica/Laboratorio-15-16-Jupyter/"
file_path = "./Datasets"

# all the three datasets contain data points on (x,y)
file_name1 = "3-clusters.csv"
file_name2 = "dataset-DBSCAN.csv"
file_name3 = "CURE-complete.csv"

data1, feature_names1, n_samples1, n_features1 = load_data(file_path, file_name1)
data2, feature_names2, n_samples2, n_features2 = load_data(file_path, file_name2)
data3, feature_names3, n_samples3, n_features3 = load_data(file_path, file_name3)

print("dataset n. 1: n samples, n features")
print(n_samples1, n_features1)

print("dataset n. 2: n samples, n features")
print(n_samples2, n_features2)

print("dataset n. 3: n samples, n features")
print(n_samples3, n_features3)

print("\n")


# PRINT GRAPHS
# import matplotlib.pyplot as plt
#
# fig = plt.figure(figsize=(8,8))
#
# ax = fig.add_subplot(111)
# fig.subplots_adjust(top=1)
# ax.set_title('Dataset n. 1 of data points')
#
# ax.set_xlabel(feature_names1[0])
# ax.set_ylabel(feature_names1[1])
#
# #plot the dataset
# plt.plot(data1[:,0], data1[:,1], '.',markersize=1)
#
# plt.show()
#
#
#
#
#
# import matplotlib.pyplot as plt
#
# fig = plt.figure(figsize=(20,10))
#
# ax = fig.add_subplot(111)
# fig.subplots_adjust(top=1)
# ax.set_title('Dataset n. 2 of data points')
#
# ax.set_xlabel(feature_names2[0])
# ax.set_ylabel(feature_names2[1])
#
# #plot the dataset
# plt.plot(data2[:,0], data2[:,1], '.', markersize=2)
#
# plt.show()
#
#
#
#
#
# import matplotlib.pyplot as plt
#
# fig = plt.figure(figsize=(8,8))
#
# ax = fig.add_subplot(111)
# fig.subplots_adjust(top=1)
# ax.set_title('Dataset n. 3 of data points')
#
# ax.set_xlabel(feature_names3[0])
# ax.set_ylabel(feature_names3[1])
#
# #plot the dataset
# plt.plot(data3[:,0], data3[:,1], '.', markersize=0.5, markeredgecolor = 'none')
#
# plt.show()

# execute a simple training of the kmeans classifier, and the plots the cluster results
# if the classifier is already trained, it is not trained anymore
def train_kmeans_and_plot(data_to_fit, k, clf_kmeans= None):

    np.random.seed(5)

    if clf_kmeans is None:
        clf_kmeans = KMeans(n_clusters=k, random_state=0).fit(data_to_fit)

    # i = 0
    # for i in range(n_samples):
    #     print("Example n." + str(i) + "=(" + str(data_to_fit[i, 0]) + "," + str(data_to_fit[i, 1]) + ")")
    #     print("in cluster n." + str(kmeans1.labels_[i]))

    # plotting
    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=1)
    ax.set_title('Clustered points in dataset')

    ax.set_xlabel('x')
    ax.set_ylabel('y')


    # set the list of colors to be selected when plotting the different clusters
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    # plot the dataset
    n_samples = len(data_to_fit)

    for clu in range(k):
        # collect the sequence of cooordinates of the points in each given cluster (determined by clu)
        data_list_x = [data_to_fit[i, 0] for i in range(n_samples) if clf_kmeans.labels_[i] == clu]
        data_list_y = [data_to_fit[i, 1] for i in range(n_samples) if clf_kmeans.labels_[i] == clu]
        plt.scatter(data_list_x, data_list_y, s=50, edgecolors='none', c=color[clu], alpha=0.5)

    plt.show()


# this method returns the best classifier for a given k and a number of attempts
def execute_kmeans_several_times(data_to_fit, k, type_of_score='silhouette', kernel=None):

    NUMBER_OF_ITERATIONS = 2

    best_score = -2
    best_k_mean_clf = None

    i = 0

    while i < NUMBER_OF_ITERATIONS:
        i += 1

        np.random.seed(5)

        # getting the classifier...
        clf = None
        if kernel is None:
            clf = KMeans(n_clusters=k, random_state=0)
        elif kernel == 'rbf':
            # use the gaussian kernel
            clf = SpectralClustering(n_clusters=k, random_state = 0, affinity='rbf')

        labels = clf.fit_predict(data_to_fit)

        # the inertia can be calculated if no kernel is applied
        if kernel is not None or type_of_score == 'silhouette':

            # silhouette score
            sil_score = silhouette_score(data_to_fit, labels)

            if best_score == -2 or sil_score > best_score:
                best_score = sil_score
                best_k_mean_clf = clf
        else:

            # inertia
            if best_score == -2 or clf.inertia_ < best_score:
                best_score = clf.inertia_
                best_k_mean_clf = clf

    return [best_score, best_k_mean_clf]


# this method returns the best classifier, using a specific type of score and a kernel
def execute_kmeans_for_different_k(data_to_fit, type_of_score='silhouette', kernel=None):

    MAX_K = 7

    scores = []

    # si parte con due clusters
    k = 2

    print("Training the kmeans algorithm, using the score of type: " + type_of_score + ("" if kernel is None else (", kernel of type: " + kernel)))

    while k <= MAX_K:

        score, k_mean_clf = execute_kmeans_several_times(data_to_fit, k, type_of_score, kernel)
        print("The score for k=" + str(k) + " is: " + str(score))

        scores.append([int(k), score, k_mean_clf])

        k+=1

    scores = np.array(scores)

    if type_of_score == 'silhouette':
        return analyze_silhouette_scores(scores, data_to_fit)
    else:
        return analyze_inertia_scores(scores, data_to_fit)


def analyze_silhouette_scores(scores, data_to_fit):

    # get best index of score
    index_best_k = np.argmax(scores[:, 1])

    # best number of clusters
    best_k = int(scores[index_best_k, 0])
    print("Best k = " + str(best_k))

    # best score
    best_score = scores[index_best_k, 1]
    print("Best score = " + str(best_score))

    # plot scores
    plt.plot(scores[:, 0], scores[:, 1], '-')
    plt.title("Silhouette scores")
    plt.show()

    # get best classifier
    best_k_mean_clf = scores[index_best_k, 2]

    # PLOT the silhouette values and the graphical clusters
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Get silhouette samples
    silhouette_vals = silhouette_samples(data_to_fit, best_k_mean_clf.labels_)

    # Plot silhouette values grouped for each cluster
    y_lower, y_upper = 0, 0
    for i, cluster in enumerate(np.unique(best_k_mean_clf.labels_)):
        cluster_silhouette_vals = silhouette_vals[best_k_mean_clf.labels_ == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
        ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
        y_lower += len(cluster_silhouette_vals)

    # Get the average silhouette score and plot it
    avg_score = np.mean(silhouette_vals)
    ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')
    ax1.set_yticks([])
    ax1.set_xlim([-0.1, 1])
    ax1.set_xlabel('Silhouette coefficient values')
    ax1.set_ylabel('Cluster labels')
    ax1.set_title('Silhouette plot for the various clusters', y=1.02);

    # Scatter plot of data colored with labels for the second graphic

    # plot centroids in case the classifier has them
    if hasattr(best_k_mean_clf, 'cluster_centers_'):
        centroids = best_k_mean_clf.cluster_centers_
        ax2.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='r', s=250)

    # plot instances
    ax2.scatter(data_to_fit[:, 0], data_to_fit[:, 1], c=best_k_mean_clf.labels_)

    ax2.set_xlim([-2, 2])
    ax2.set_xlim([-2, 2])
    ax2.set_title('Visualization of clustered data', y=1.02)
    ax2.set_aspect('equal')
    plt.tight_layout()
    plt.suptitle(f'Silhouette analysis using k = {best_k}',
                 fontsize=16, fontweight='semibold', y=1.05)

    plt.show()

    return best_k_mean_clf


def analyze_inertia_scores(scores, data_to_fit):

    # the best k is chosen using the elbow method
    # for now there isn't a sistematic method to chose the best k
    plt.plot(scores[:, 0], scores[:, 1], '-')
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia value")
    plt.ylim([0, scores[0, 1] * 1.1])
    plt.title("Pick the best k")
    plt.show()

    print("Specify the best k:")
    input_k = int(input())

    input_clf = scores[input_k-2, 2]

    train_kmeans_and_plot(data_to_fit, input_k, input_clf)

    return input_clf


def execute_DBSCAN_and_plot(data_to_fit, eps=0.5, min_samples=5):

    np.random.seed(5)

    clf_dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(data_to_fit)

    # set the list of colors to be selected when plotting the different clusters
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    # plot the dataset
    n_samples = len(data_to_fit)

    n_clusters = max(set(clf_dbscan.labels_)) + 1

    print("n_clusters=" + str(n_clusters))

    for clu in range(n_clusters):
        # collect the sequence of cooordinates of the points in each given cluster (determined by clu)
        data_list_x = [data_to_fit[i, 0] for i in range(n_samples) if clf_dbscan.labels_[i] == clu]
        data_list_y = [data_to_fit[i, 1] for i in range(n_samples) if clf_dbscan.labels_[i] == clu]
        plt.scatter(data_list_x, data_list_y, s=20, edgecolors='none', c=color[clu], alpha=0.5)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Learned clusters using DBSCAN (eps=" + str(eps) + ", minPts=" + str(min_samples) + ")\n Number of clusters: " + str(n_clusters))
    plt.show()


# for a large array, is better to use this kind of sorting
def efficient_selection_sort(array, max_n):

    for i in range(max_n):

        swap = i + np.argmin(array[i:])
        (array[i], array[swap]) = (array[swap], array[i])

    return array[:max_n]


def plot_reachability_distance(data, k_to_consider):

    # create matrix of distances
    print("Creating matrix of distances...")
    data = np.array(data)

    N = len(data)
    max_k = k_to_consider[-1] + 1

    # compute distances with other points and sort them
    # take only the distances needed for the analisys
    distances = np.empty((N, max_k), dtype=type(data[0, 0]))

    index = 0
    while index < N:

        # when max_k is small, it is better to use the selection sort

        distances[index] = np.sort(np.linalg.norm(np.subtract(data, data[index]), axis=1))[:max_k]
        #distances[index] = efficient_selection_sort(np.linalg.norm(np.subtract(data, data[index]), axis=1), max_k)

        index += 1
        #if index % 128 == 0:
            #print("Computing data number -> " + str(index))

    # per ogni k, consideriamo il valore che deve avere epsilon affinché ci sia il numero massimo di core points,
    # senza però che epsilon sia troppo grande per creare dei cluster non significativi
    # il valore di epsilon si ferma dopo che la distanza tra il k-esimo vicino inizia ad aumentare di tanto (valore empirico)
    best_k = None
    best_eps = None
    max_points = 0

    for k in range(k_to_consider[0], max_k):

        # get the k-nearest distances for each point
        k_nearest_distances = np.sort(distances[:, k])

        # calcolo fino a quando è necessario incrementare eps

        # valore iniziale
        n_points = int(N / 100)
        eps = k_nearest_distances[n_points - 1]

        while n_points < N:

            # ottengo la distanza
            value = k_nearest_distances[n_points]

            if value > eps:

                # se il valore è troppo grande rispetto al precedente, allora mi fermo con l'incrementare epsilon
                if value > eps * 1.01:
                    break
                else:
                    eps = value

            n_points += 1

        # take the best eps and n_points in order to cluster much data as possible
        if n_points > max_points:
            max_points = n_points
            best_k = k
            best_eps = eps

        if k in k_to_consider:
            # plot the rechability distance
            plt.plot(range(0, N), k_nearest_distances, '-')
            plt.xlabel("Instances")
            plt.ylabel("Distance")
            plt.title("Rechability distance for the " + str(k) + "-nearest neighbour")
            plt.show()

    del distances

    print("The best k is " + str(best_k) + ", with the epsilon value of " + str(best_eps))
    return [best_eps, best_k]


# standardize data
data1 = StandardScaler().fit_transform(data1)
data2 = StandardScaler().fit_transform(data2)
data3 = StandardScaler().fit_transform(data3)

# KMEANS

# use the default classifier
print("\nUsing kmeans for the first dataset")
execute_kmeans_for_different_k(data1)

# using the Radial basis function kernel for the second kernel
print("\nUsing kmeans for the second dataset")
execute_kmeans_for_different_k(data2, kernel='rbf')

# the silhouette score for the third dataset is too expensive, the inertia score will be used
print("\nUsing kmeans for the third dataset")
execute_kmeans_for_different_k(data3, 'inertia')


# DBSCAN

# get the best eps and minPts for the dataset
print("\nUsing DBSCAN for the second dataset")
eps, minPts = plot_reachability_distance(data2, [4, 5, 10, 30])
execute_DBSCAN_and_plot(data2, eps, minPts)

# the third dataset is too big for the computer, we must optimize the resources
print("\nUsing DBSCAN for the third dataset")
# data3 = np.float16(data3)

# get the best eps and minPts for the dataset
# eps, minPts = plot_reachability_distance(data3, [4, 5, 10, 20, 50, 100, 200, 300, 400, 600, 800, 1000, 1300, 1700])
# execute_DBSCAN_and_plot(data3, eps, minPts)
execute_DBSCAN_and_plot(data3, 0.085, 200)

