'''
Additional classifier questions
'''
import numpy as np

#############################################
########### Naive Bayes Questions ###########
#############################################

def question_nb_a(clf, train_features, train_labels):
    # This fits the model to the training data. Ensure your implementation for
    # `NaiveBayes.fit()` is correct!
    clf.fit(train_features, train_labels)

    epsilon = 1e-8 # prevents division by zero in case `fit()` is not implemented
    res = clf.label_counts[1] / ((clf.label_counts[0] + clf.label_counts[1]) or epsilon)
    return res

def question_nb_b(clf, train_features, train_labels):
    clf.fit(train_features, train_labels)
    res = np.zeros(train_features.shape[1])

    ### YOUR CODE HERE

    ### END YOUR CODE

    return res

def question_nb_c(clf, train_features, train_labels):
    clf.fit(train_features, train_labels)
    res = np.zeros(train_features.shape[1])

    ### YOUR CODE HERE

    ### END YOUR CODE

    return res

def question_nb_d(clf, train_features, train_labels):
    clf.fit(train_features, train_labels)
    res = np.zeros(5, np.uint8)

    ### YOUR CODE HERE

    ### END YOUR CODE

    return res