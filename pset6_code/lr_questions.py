'''
Additional classifier questions
'''
import numpy as np

#############################################
####### Logistic Regression Questions #######
#############################################

def question_lr_a(clf, train_features, train_labels):
    clf.fit(train_features, train_labels)
    res = np.zeros(5, np.uint8)

    ### YOUR CODE HERE

    ### END YOUR CODE

    return res

def question_lr_b(clf, train_features, train_labels, test_features, test_labels):
    clf.fit(train_features, train_labels)
    res = 0.0

    ### YOUR CODE HERE

    ### END YOUR CODE

    return res

def question_lr_c(clf, train_features, train_labels, test_features, test_labels):
    clf.fit(train_features, train_labels)
    res = 0.0

    ### YOUR CODE HERE

    ### END YOUR CODE

    return res
