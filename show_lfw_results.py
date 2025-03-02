import matplotlib.pyplot as plt
import numpy as np
import activations as act
import utils
from time import time

# load data from csv file
# (widths, one_layer_accuracies, one_layer_std, one_layer_probs, two_layer_accuracies, two_layer_std, two_layer_probs)
results = np.loadtxt('lfw_results_400.csv')
widths = results[:, 0]
one_layer_accuracies, one_layer_std, one_layer_probs = results[:, 1], results[:, 2], results[:, 3]
two_layer_accuracies, two_layer_std, two_layer_probs = results[:, 4], results[:, 5], results[:, 6]

# accuracies plot
opacity=0.2

plt.plot(widths, one_layer_accuracies, label="One Layer")
plt.fill_between(widths,one_layer_accuracies+one_layer_std,
        one_layer_accuracies-one_layer_std, alpha=opacity)

plt.plot(widths, two_layer_accuracies, label="Two Layers")
plt.fill_between(widths,two_layer_accuracies+two_layer_std,
        two_layer_accuracies-two_layer_std, alpha=opacity)

plt.title('Average Accuracy')
plt.xlabel('Width')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# probabilities plot

plt.plot(widths, one_layer_probs, label="One Layer")
plt.plot(widths, two_layer_probs, label="Two Layers")

plt.title('Separation Probability')
plt.xlabel('Width')
plt.ylabel('Probability')
plt.legend()
plt.show()