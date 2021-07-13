import numpy as np
from matplotlib import pyplot as plt


def apply_linear_model(model, data):
    return np.dot(data, np.transpose(model)) > 0


def build_confusion_matrix(predicted_labels, target_labels):

    i = 0

    tp = 0.
    tn = 0.
    fp = 0.
    fn = 0.

    while i < len(predicted_labels):
        if predicted_labels[i] == target_labels[i]:
            if target_labels[i] == 0:
                tn += 1
            else:
                tp += 1
        else:
            if target_labels[i] == 0:
                fp += 1
            else:
                fn += 1
        i += 1

    return [tp, tn, fp, fn]


# calculate accuracy: (TP + TN) / N samples
def calculate_accuracy(confusion_matrix):
    return (confusion_matrix[0] + confusion_matrix[1]) / 1000


# calculate proximity to rock heaven: TP - FP
def calculate_coverage_heaven(confusion_matrix):
    return confusion_matrix[0] - confusion_matrix[2]


NUMBER_OF_RANDOM_MODELS = 100

# generate random data
data = np.random.random_integers(-100, 100, [1000, 2])
print(data)

# create a new linear model
target_model = [4., -1.]
target_labels = apply_linear_model(target_model, data)

# plot points of the new linear model
colors = ['r' if l else 'b' for l in target_labels]
plt.scatter(data[:, 0], data[:, 1], color=colors)
plt.show()

# create 100 random linear models
models = (np.random.rand(NUMBER_OF_RANDOM_MODELS, 2) - 0.5) * 10

# (ex. 1) for each model, build the confusion matrix
confusion_matrixes = []
i = 0

for model in models:

    # predict labels by this model
    predicted_labels = apply_linear_model(model, data)

    # create confusion matrix
    conf_matrix = build_confusion_matrix(predicted_labels, target_labels)
    confusion_matrixes.append(conf_matrix)

    i += 1

# print confusion matrixes
confusion_matrixes = np.array(confusion_matrixes)
print("Confusion matrixes (tp, tn, fp, fn)\n", confusion_matrixes, "\n")

# (ex. 2) for each model, plot the [FP,TP] pairs on a scatter plot
plt.scatter(confusion_matrixes[:, 2], confusion_matrixes[:, 0])
plt.xlabel('FP')
plt.ylabel('TP')
plt.show()

# (ex. 3) the best model (considering the accuracy) is top-left corner of the plot

# calculate accuracy for each model
result = list(map(calculate_accuracy, confusion_matrixes))

best_model_index = np.argmax(result)
print("Best accuracy model:\n", models[best_model_index])
print("The confusion matrix is: \n", confusion_matrixes[best_model_index], "\n")

# (ex. 4) the model with the best accuracy is close to the target model. It is the one nearest to the top-left corner

# calculating the model nearest to the top-left corner
result = list(map(calculate_coverage_heaven, confusion_matrixes))

best_model_index = np.argmax(result)
print("Best model based on the coverage plot:\n", models[best_model_index])
print("The confusion matrix is: \n", confusion_matrixes[best_model_index])

# (ex 5) Dato che sono stati generati 100 modelli nel range [-5, 5], è plausibile che almeno uno di essi si avvicini al modello target
