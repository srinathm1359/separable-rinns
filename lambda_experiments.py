#Test the performance of a neural network with varying lambda.
import matplotlib.pyplot as plt
import numpy as np
import activations as act
from sklearn import svm
import utils

#parameters
name = '20'
SHOW_DATASET = False
num_trials = 1000
bias_lb = 0
bias_ub = 500
num_samples = 200
width = 30

#load pre-processed data
points, labels, Y, radius = utils.process_data(name)

def get_shallow_net_performance(num_trials, width, bias, depth, bad_method=False):
    """
    Given a width and maximal bias, returns the average performance (accuracy and the probability of 100% accuracy)
    after averaging over num_trials times.

    Args:
        num_trials: int, should be at least 1
        width: int, should be at least 1
        bias: float or int, should be at least 0
        depth: int, should be 1 or 2
    """
    trials = np.zeros(num_trials)
    successes = np.zeros(num_trials)
    for j in range(num_trials):
        if depth == 1:
            X = act.one_layer_relu_net(points, width, bias)
        elif depth == 2:
            X = act.one_layer_relu_net(points, width, bias)
            if bad_method:
                X = act.one_layer_relu_net(X, width, bias)
            else:
                next_bias = np.sqrt(bias**2/3 + radius**2)
                X = act.one_layer_relu_net(X, width, next_bias)
        linear = svm.LinearSVC(dual=False, C=0.3, max_iter=100000).fit(X, Y)
        trials[j] = linear.score(X,Y)
        if trials[j] == 1:
            successes[j] = 1
    score = np.mean(trials)
    prob = np.mean(successes)
    return score, prob

def run_lambda_experiment(num_trials, bias_lb, bias_ub, width, depth, num_samples, bad_method=False):
    """
    Computes the average performance of a randomly initialized neural network as the maximal bias varies.
    Args:
        num_trials: Number of trials used to estimate performance.
        bias_lb: Lower bound on the range of maximal biases we test.
        bias_ub: Upper bound on the range of maximal biases we test.
        width: Width of the randomly initialized neural network.
        depth: Depth of the randomly initialized neural network.
    """
    biases = np.linspace(bias_lb,bias_ub,num_samples)
    scores = []
    probs = []
    for bias in biases:
        score, prob = get_shallow_net_performance(num_trials, width, bias, depth, bad_method)
        print('Lambda ' + str(bias) + ' has average score ' + str(score) + ' and separation probability ' + str(prob))
        scores.append(score)
        probs.append(prob)
    return biases, scores, probs

if __name__ == "__main__":
    if SHOW_DATASET:
        utils.plot_points(points,labels, title="Original Dataset")
        
    results = []
    depths = [1,2]

    for depth in depths:
        print("Testing depth", depth)
        results.append(run_lambda_experiment(num_trials, bias_lb, bias_ub, width, depth, num_samples))

    print("Testing bad method, depth", 2)
    results.append(run_lambda_experiment(num_trials, bias_lb, bias_ub, width, 2, num_samples, bad_method=True))

    for i in range(len(depths)):
        depth = depths[i]
        biases, scores, probs = results[i]
        plt.plot(biases, scores, label=('Depth = ' + str(depth)))
    
    biases, scores, probs = results[-1]
    plt.plot(biases, scores, label=('Depth 2, suboptimal bias'))
    
    plt.title('Average Accuracy')
    plt.xlabel('Maximal Bias')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    for i in range(len(depths)):
        depth = depths[i]
        biases, scores, probs = results[i]
        plt.plot(biases, probs, label=('Depth = ' + str(depth)))

    biases, scores, probs = results[-1]
    plt.plot(biases, probs, label=('Depth 2, suboptimal bias'))

    plt.title('Separation Probability')
    plt.xlabel('Maximal Bias')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()