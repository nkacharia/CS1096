'''
Naive Bayes Classifier
'''
import numpy as np

class NaiveBayes:
    '''
    Naive Bayes Classifier (Bernoulli event model)

    During training, the classifier learns probabilities by counting the
    occurences of feature/label combinations that it finds in the
    training data. During prediction, it uses these counts to
    compute probabilities.
    '''

    def __init__(self, use_laplace_add_one):
        self.label_counts = {}
        self.feature_counts = {}
        self.use_laplace_add_one = use_laplace_add_one # True for Laplace add-one smoothing

    def fit(self, train_features, train_labels):
        '''Training stage - learn from data'''

        self.label_counts[0] = 0
        self.label_counts[1] = 0

        ### YOUR CODE HERE (~5-10 Lines)

        ### END YOUR CODE

    def predict(self, test_features):
        '''Testing stage - classify new data'''

        preds = np.zeros(test_features.shape[0], dtype=np.uint8)

        ### YOUR CODE HERE (~10-30 Lines)

        ### END YOUR CODE

        return preds
