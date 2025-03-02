# We test how well using a random neural network to separate the data works in a real-world dataset.
# We will measure the performance on the entire dataset to test its separation capacity.
# Dataset: http://vis-www.cs.umass.edu/lfw/#download

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
import activations as act
import utils
from time import time
from sklearn import svm
from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.utils.fixes import loguniform
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

start = time()
print("Fetching data...")
lfw_people = fetch_lfw_people(min_faces_per_person=50, resize=0.4)
fetch_time = time()
print("Fetched data in", fetch_time-start, "seconds")


# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape
print("Image shape:", (h,w))

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# We are not using any test data, so we can just test the separation capacity
X_train, y_train = X, y

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# calculate radius of dataset to inform the bias parameters
magnitudes = np.linalg.norm(X_train, axis=1)
radius = np.amax(magnitudes)
Lambda = radius
Lambda_2 = np.sqrt(radius**2 + Lambda**2 / 3)

def compute_accuracy(random_state, width, features_type='none'):
    np.random.seed(random_state)
    # features
    if features_type == 'none':
        X_final = X_train
    elif features_type == 'one_layer':
        X_final = act.one_layer_relu_net(X_train, width, Lambda)
    elif features_type == 'two_layer':
        X_final = act.one_layer_relu_net(X_train, width, Lambda)
        X_final = act.one_layer_relu_net(X_final, width, Lambda_2)

    clf = LinearSVC(dual=False,C=1e3,class_weight="balanced", random_state=random_state, max_iter=1e3) # make predictions
    clf = clf.fit(X_final, y_train)
    return clf.score(X_final,y_train) # training accuracy

def do_one_trial(random_state, width, features_type):
    accuracy = compute_accuracy(random_state, width, features_type=features_type)
    prob = 0
    if accuracy == 1:
        prob = 1
    return accuracy, prob

def fixed_width_test(width, num_trials):
    """
    Fix the width and test the performance of one-layer vs. two-layer randomly initialized networks.
    """
    # Tests using one layer
    random_states = [i for i in range(num_trials)]
    one_accuracies, one_probs = [], []
    for random_state in random_states:
        accuracy, prob = do_one_trial(random_state, width, features_type='one_layer')
        one_accuracies.append(accuracy)
        one_probs.append(prob)
    one_accuracies = np.array(one_accuracies)
    one_probs = np.array(one_probs)
    one_layer_results = (one_accuracies.mean(), one_accuracies.std(), one_probs.mean())
    
    # Tests using two layers
    two_accuracies, two_probs = [], []
    for random_state in random_states:
        accuracy, prob = do_one_trial(random_state, width, features_type='two_layer')
        two_accuracies.append(accuracy)
        two_probs.append(prob)
    two_accuracies = np.array(two_accuracies)
    two_probs = np.array(two_probs)
    two_layer_results = (two_accuracies.mean(), two_accuracies.std(), two_probs.mean())
    return one_layer_results, two_layer_results

# Running the experiment
num_trials = 100
widths = np.arange(10, 401, 10)
one_layer_accuracies, one_layer_std, one_layer_probs = [], [], []
two_layer_accuracies, two_layer_std, two_layer_probs = [], [], []

for width in widths:
    start = time()

    one_layer_results, two_layer_results = fixed_width_test(width, num_trials)
    one_acc, one_std, one_prob = one_layer_results
    two_acc, two_std, two_prob = two_layer_results

    one_layer_accuracies.append(one_acc)
    one_layer_std.append(one_std)
    one_layer_probs.append(one_prob)

    two_layer_accuracies.append(two_acc)
    two_layer_std.append(two_std)
    two_layer_probs.append(two_prob)
    
    print("Completed width", width, "in", time() - start, "seconds")
    
one_layer_accuracies = np.array(one_layer_accuracies)
one_layer_std = np.array(one_layer_std)
one_layer_probs = np.array(one_layer_probs)

two_layer_accuracies = np.array(two_layer_accuracies)
two_layer_std = np.array(two_layer_std)
two_layer_probs = np.array(two_layer_probs)

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

# save the data into a txt file
file_name = "lfw_results_400.csv"
result = (widths, one_layer_accuracies, one_layer_std, one_layer_probs, two_layer_accuracies, two_layer_std, two_layer_probs)
data = np.empty((widths.shape[0], 7))
for i in range(7):
    data[:, i] = result[i]
np.savetxt(file_name, data, delimiter=" ")