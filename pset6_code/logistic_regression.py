'''
Logistic Regression Classifier
'''
from time import time
import numpy as np

class LogisticRegression:
    '''
    Logistic Regression Classifier

    During training, Logistic Regression learns weights for each
    feature using gradient ascent. During prediction, it uses
    the test data to apply a linear transformation to the weights,
    obtaining a probability for each example in the test data.
    '''

    def __init__(self, learning_rate, max_steps):
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.weights = None

    def fit(self, train_features, train_labels):
        '''Training stage - learn from data'''

        # This line inserts a column of ones before the first column of train_features,
        # resulting in the an `n x (d + 1)` size matrix, This is so we
        # don't need to have a special case for the bias weight.
        train_features = np.insert(train_features, 0, 1, axis=1)

        # This makes the matrix immutable
        train_features.setflags(write=False)

        # This is the theta you will be performing gradient ascent on. It has
        # shape (d + 1).
        theta = np.zeros(train_features.shape[1])

        ### YOUR CODE HERE (~3-10 Lines)

        ### END YOUR CODE

        self.weights = theta

    def predict(self, test_features):
        '''Testing stage - classify new data'''

        test_features = np.insert(test_features, 0, 1, axis=1) # add bias term
        test_features.setflags(write=False) # make immutable
        preds = np.zeros(test_features.shape[0], np.uint8)

        ### YOUR CODE HERE (~1-7 Lines)

        ### END YOUR CODE

        return preds

def sigmoid(vec):
    '''Numerically stable implementation of the sigmoid function'''
    positive_mask = vec >= 0
    negative_mask = vec < 0
    exp = np.zeros_like(vec, dtype=np.float64)
    exp[positive_mask] = np.exp(-vec[positive_mask])
    exp[negative_mask] = np.exp(vec[negative_mask])
    top = np.ones_like(vec, dtype=np.float64)
    top[negative_mask] = exp[negative_mask]
    return top / (1 + exp)

class Stopwatch:
    '''
    Display loop progress

    Wrap an iterable object, and obtain automatic updates on the progress of iteration.

    Example:
        Replace `for i in range(15):` with `for i in Stopwatch(range(15)):`
        That's it! The `Stopwatch` will display information about the progress of the loop.
    '''

    def __init__(self, iterable):
        self.iterable = tuple(iterable)

    def __iter__(self):
        start_time = time()
        num_iters = len(self.iterable)
        for i, val in enumerate(self.iterable):
            yield val
            cur_time = time()
            elapsed = cur_time - start_time
            i = i + 1 #consider having already completed the iteration
            print(
                f'Progress: {i / num_iters * 100:.2f}%, '
                f'Elapsed: {elapsed:.2f} sec, '
                f'Estimated remaining: {elapsed * (num_iters - i) / i:.2f} sec',
                end='\r'
            )
        print()
