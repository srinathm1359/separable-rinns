#Test performance of a neural network with varying width.
import matplotlib.pyplot as plt
import numpy as np
import activations as act
import utils
from sklearn import svm

#parameters
name = '20'
SHOW_DATASET = False
num_trials = 1000
width_lb = 1
width_ub = 81

#load pre-processed data
points, labels, Y, radius = utils.process_data(name)

Lambda = radius
Lambda_2 = np.sqrt(radius**2 + Lambda**2 / 3)

def get_shallow_net_performance(num_trials, width, depth):
    """
    Given a width and depth, returns the average performance (accuracy and the probability of 100% accuracy)
    after averaging over num_trials times.

    Args:
        num_trials: int, should be at least 1
        width: int, should be at least 1
        depth: int, only 1 or 2
    """
    trials = np.zeros(num_trials)
    successes = np.zeros(num_trials)
    for j in range(num_trials):
        if depth == 1:
            X = act.one_layer_relu_net(points, width, Lambda)
        elif depth == 2:
            X = act.one_layer_relu_net(points, width, Lambda)
            X = act.one_layer_relu_net(X, width, Lambda_2)
        linear = svm.LinearSVC(dual=False, C=0.3, max_iter=100000).fit(X, Y)
        trials[j] = linear.score(X,Y)
        if trials[j] == 1:
            successes[j] = 1
    score = np.mean(trials)
    prob = np.mean(successes)
    print('Width ' + str(width) + ' has average score ' + str(score) + ' and separation probability ' + str(prob))
    return score, prob

def run_shallow_net_experiment(num_trials, width_lb, width_ub, depth):
    """
    Computes the average performance of a randomly initialized neural network as width varies.
    Args:
        num_trials: Number of trials used to estimate performance.
        width_lb: Lower bound on the range of widths we test.
        width_ub: Upper bound on the range of widths we test.
        depth: Depth of the randomly initialized neural network.
    """
    widths = np.arange(width_lb,width_ub,1)
    scores = []
    probs = []
    for width in widths:
        score, prob = get_shallow_net_performance(num_trials, width, depth)
        scores.append(score)
        probs.append(prob)
    return widths, scores, probs

if __name__ == "__main__":
    if SHOW_DATASET:
        utils.plot_points(points,labels, title="Original Dataset")

    results = []
    depths = [1,2]

    for depth in depths:
        print("Testing depth", depth)
        results.append(run_shallow_net_experiment(num_trials,width_lb,width_ub, depth))

    for i in range(len(depths)):
        depth = depths[i]
        widths, scores, probs = results[i]
        plt.plot(widths, scores, label=('Depth = ' + str(depth)))
    plt.title('Average Accuracy')
    plt.xlabel('Width')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    for i in range(len(depths)):
        depth = depths[i]
        widths, scores, probs = results[i]
        plt.plot(widths, probs, label=('Depth = ' + str(depth)))
    plt.title('Separation Probability')
    plt.xlabel('Width')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()