import numpy as np
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score  # will be used to get accuracy scores
from scipy import stats # used for getting the critical value

class_labels = ["Joy", "Sadness"]
n_features = 11288  # number of columns in the matrix = number of features (distinct elements in the documents)
n_rows = 11981  # number rows of the matrix
n_elements = 71474  # number of the existing values in the matrix (not empty, to be loaded in the matrix in a sparse way)

# path_training="/Users/meo/Documents/Didattica/Laboratorio-15-16-Jupyter/"
path_training = "Datasets for Naive Bayes Classification/"
file_name = "joy_sadness6000.txt"

# declare the row and col arrays with the indexes of the matrix cells (non empty) to be loaded from file
# they are needed because the matrix is sparse and we load in the matrix only the elements which are present
row = np.empty(n_elements, dtype=int)
col = np.empty(n_elements, dtype=int)
data = np.empty(n_elements, dtype=int)

row_n = 0  # number of current row to be read and managed
cur_el = 0  # position in the arrays row, col and data
twitter_labels = []  # list of class labels (target array) of the documents (twitter) that will be read from the input file
twitter_target = []  # list of 0/1 for class labels

with open(path_training + file_name, "r") as fi:
    for line in fi:
        el_list = line.split(',')
        l = len(el_list)

        last_el = el_list[l-1]  # I grab the last element in the list which is the class label
        class_name = last_el.strip()  # eliminate the '\n'
        twitter_labels.append(class_name)
        # twitter_labels contains the labels (Joy/Sadness); twitter_target contains 0/1 for the respective labels
        if class_name == class_labels[0]:
           twitter_target.append(0)
        else:
           twitter_target.append(1)

        i = 0  # I start reading all the doc elements from the beginning of the list
        while i < (l-1):
            element_id = int(el_list[i])  # identifier of the element in the document
            element_id = element_id-1  # the index starts from 0 (the read id starts from 1)
            i = i+1
            value_cell = int(el_list[i])  # make access to the following value in the file which is the count of the element in the documento
            i = i+1

            row[cur_el] = row_n  # load the data in the three arrays: the first two are the row and col indexes; the last one is the matrix cell value
            col[cur_el] = element_id
            data[cur_el] = value_cell

            cur_el = cur_el+1
        row_n = row_n+1
fi.close()

#print("final n_row=" + str(row))

# loads the matrix by means of the indexes and the values in the three arrays just filled
twitter_data = csr_matrix((data, (row, col)), shape=(n_rows, n_features)).toarray()

'''
print("resulting matrix:")
print(twitter_data)
print(twitter_labels)
print(twitter_target)
'''

# get training and test sets
np.random.seed()

X_train, X_test, y_train, y_test = train_test_split(twitter_data, twitter_target, test_size=0.40)


# TRAIN CLASSIFIERS (ex 1)

# MULTINOMIAL MODEL
clf_multinomial = MultinomialNB(alpha=1, fit_prior=True)
#clf_multinomial.fit(X_train, y_train)

# MULTIVARIATE MODEL
clf_multivariate = BernoulliNB(alpha=1, fit_prior=True)
#clf_multivariate.fit(X_train, y_train)

# Accuracy scores (ex 2)
def execute_paired_t_test(values, alpha):

    print("The difference scores are:\n", values)

    # number of folds
    N = len(values)

    print("Running paired t-test, with " + str(n_folds) + " folds, with alpha = " + str(alpha))

    sum_el = sum(values)  # sum of elements
    squared_sum = sum_el ** 2  # squared sum of elements
    sum_squares = sum(values ** 2)  # sum of squared elements

    # calculate the t-score
    t_score = sum_el / np.sqrt((sum_squares * N - squared_sum) / (N - 1))

    print("The t-score is: " + str(t_score), '\n')

    # two tailed critical value
    critical_value = stats.t.ppf(1 - alpha / 2, N - 1)

    print("The critical value is: " + str(critical_value))

    # return result (0 if the null hypothesis is true)
    if abs(t_score) > critical_value:
        return np.sign(sum_el)
    else:
        return 0


# number of folds
n_folds = 10

# significance level
alpha = 0.05

# get classifiers' accuracy scores
scores_multinomial = np.array(cross_val_score(clf_multinomial, X_test, y_test, cv=n_folds))
scores_multivariate = np.array(cross_val_score(clf_multivariate, X_test, y_test, cv=n_folds))

print("Scores for the multinomial classifier:\n", scores_multinomial)
print("Scores for the multivariate classifier:\n", scores_multivariate)
print()

# execute paired t-test
difference_scores = scores_multinomial - scores_multivariate
result = execute_paired_t_test(difference_scores, alpha)

# print results
if result == 0:
    print("The two classifiers don't present differences in the accuracy scores")
else:
    print("The " + ("first" if result == 1 else "second") + " classifier if better than the other one")




'''
multivariate_accuracy = clf_multivariate.score(X_test, y_test)
multinomial_accuracy = clf_multinomial.score(X_test, y_test)
relative_difference = abs(multivariate_accuracy - multinomial_accuracy) * 100 / max(multinomial_accuracy, multivariate_accuracy)

print("Multivariate accuracy: " + str(multivariate_accuracy))
print("Multinomial accuracy: " + str(multinomial_accuracy))
print("Relative difference (%) in accuracy between models: " + str(relative_difference))
'''