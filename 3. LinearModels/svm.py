import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


# define input data
X = np.array([[ 0.46613554,  0.92048757],
       [-0.92129195,  0.06723639],
       [-0.15836636,  0.00430243],
       [-0.24055905, -0.87032292],
       [ 0.06245105, -0.53698416],
       [-0.2265037 , -0.43835751],
       [-0.00480479, -0.17372081],
       [-0.1525277 , -0.34399658],
       [-0.27360329,  0.35339202],
       [-0.77464508, -0.48715511],
       [-0.58724291,  0.74419972],
       [-0.97596949, -0.72172963],
       [ 0.42376225, -0.72655597],
       [ 0.96383922, -0.23371331],
       [ 0.16264643, -0.46949742],
       [-0.74294705, -0.42576417],
       [ 0.05089437, -0.20522071],
       [-0.19442744,  0.09617478],
       [-0.97102743,  0.79663992],
       [ 0.0596995 , -0.70129219],
       [-0.83934851, -0.95616033],
       [-0.38249705,  0.4973605 ],
       [ 0.3474666 ,  0.70664397],
       [ 0.35871444,  0.88679345],
       [-0.05914582,  0.23124686],
       [-0.52156643,  0.32986941],
       [-0.53579646,  0.67530208],
       [ 0.13683914, -0.96158184],
       [ 0.65904541, -0.12015303],
       [-0.69078363,  0.5615536 ],
       [ 0.47738323, -0.70919275],
       [ 0.93069669,  0.44019132],
       [ 0.19750088, -0.68869404],
       [-0.75048675, -0.18170522],
       [-0.45288395, -0.25894991],
       [-0.74644547,  0.87781953],
       [ 0.14620452,  0.56864508],
       [ 0.25719272, -0.58405476],
       [ 0.87149524,  0.01384224],
       [-0.71473576,  0.31568314],
       [-0.252637  , -0.67418371],
       [ 0.24718308,  0.95191416],
       [-0.38149953, -0.64066291],
       [-0.23112698,  0.04678807],
       [ 0.72631766,  0.7390158 ],
       [-0.91748062, -0.15131021],
       [ 0.74957917,  0.66966866],
       [ 0.76771849,  0.06662777],
       [-0.04233756, -0.91320835],
       [ 0.63840333,  0.06277738],
       [-0.78887281, -0.90311183],
       [-0.73099834, -0.69587363],
       [-0.50947652, -0.99144951],
       [ 0.14294609,  0.5474932 ],
       [ 0.4367906 ,  0.31953258],
       [-0.13970851,  0.81817884],
       [ 0.6440873 ,  0.79118775],
       [ 0.41714043, -0.66672029],
       [ 0.59283022, -0.71836746],
       [ 0.55379696,  0.98846202],
       [-0.91819517,  0.34203895],
       [ 0.02020188,  0.83696694],
       [ 0.6182918 ,  0.04254014],
       [-0.09354765, -0.30050483],
       [-0.08489545,  0.06431463],
       [-0.11886358, -0.68738895],
       [ 0.44428375,  0.18273761],
       [ 0.26486362, -0.98398013],
       [ 0.13222452,  0.91495035],
       [-0.11101656,  0.00541343],
       [-0.07696178, -0.92720555],
       [ 0.22602214,  0.56040092],
       [ 0.74227542,  0.32930104],
       [ 0.43524657,  0.35332933],
       [-0.89277607, -0.59996171],
       [-0.94836212,  0.78777302],
       [ 0.1783319 , -0.2142071 ],
       [-0.07832238, -0.25046584],
       [ 0.17611799, -0.96927832],
       [-0.95938454, -0.26504646],
       [ 0.58666766, -0.94620881],
       [-0.77336565,  0.46735057],
       [-0.94414054,  0.39044333],
       [ 0.61524645,  0.15907662],
       [-0.09855302,  0.9816656 ],
       [ 0.53937097,  0.34487634]])

# define target data
y = [1 if x + y > 0.3 else -1 for [x, y] in X]

# EXERCISE


# this function trains and displays an SVC classifier
def train_and_display_SVC_classifier(clf, classifier_title):

       # train the classifier
       clf.fit(X, y)

       print(" == " + classifier_title + " == ")
       print("Number of support vectors: " + str(len(clf.support_vectors_)))
       print("Number of margin errors: " + str(sum([True if abs(coef) == clf.C else False for coef in clf.dual_coef_[0]])))

       # predict target values from training data
       y_predicted = [1 if x > 0 else -1 for x in clf.predict(X)]

       # calculate number of errors and print it
       errors = sum([True if y[i] != y_predicted[i] else False for i, w in enumerate(y)])
       print("Number of prediction errors: " + str(errors), '\n')

       # calculate colors for the support vectors
       y_dual = []

       i = 0
       while i < len(clf.dual_coef_[0]):

           x = clf.dual_coef_[0][i]

           if x == clf.C:
               y_dual.append("red")
           elif x >= 0:
               y_dual.append("orange")
           elif x == -clf.C:
               y_dual.append("green")
           else:
               y_dual.append("lightblue")

           i += 1

       # plot the data instances
       plt.scatter(X[:, 0], X[:, 1], c=y)

       # plot support vectors, differentiated by colors
       plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], c=y_dual, s=8)

       plt.title(classifier_title)
       # plt.show()



# create SVC classifiers
linear_clf = SVC(kernel='linear')
high_C_clf = SVC(kernel='linear', C=1000)
low_C_clf = SVC(kernel='linear', C=0.3)
rbf_clf = SVC(kernel='rbf', gamma="scale")

plt.subplot(2,2,1)
train_and_display_SVC_classifier(linear_clf, "Linear SVC")

# the number of support vectors is lower, and there are no margin errors
plt.subplot(2,2,2)
train_and_display_SVC_classifier(high_C_clf, "Linear SVC (C = " + str(high_C_clf.C) + ")")

# there are more support vectors and margin errors
plt.subplot(2,2,3)
train_and_display_SVC_classifier(low_C_clf, "Linear SVC (C = " + str(low_C_clf.C) + ")")

# if we use the radial basis function kernel, the classifier can learn the linear model
plt.subplot(2,2,4)
train_and_display_SVC_classifier(rbf_clf, "Rbf SVC")

plt.show()
