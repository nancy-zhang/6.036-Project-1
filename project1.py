from string import punctuation, digits

import numpy as np
import matplotlib.pyplot as plt

### Parameters
T_perceptron = 5

T_avperceptron = 5

T_avgpa = 5
L_avgpa = 10
###

### Part I

def hinge_loss(feature_matrix, labels, theta, theta_0):
    """
    Section 1.2
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """

    feature_matrix = np.matrix(feature_matrix)
    loss_h = 1-np.multiply((np.dot(feature_matrix, np.transpose(theta))+theta_0), labels)

    zeros = np.matrix(np.zeros(feature_matrix.shape[0]))
    loss_mat = np.append(zeros, loss_h, axis = 0)

    #loss_mat is a matrix with the first row all 0's. Compare within column for max(0, 1-yxtheta)
    avg_hinge_loss = np.mean(np.amax(loss_mat, axis=0))
    return avg_hinge_loss

def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    """
    Section 1.3
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """

    theta_update = np.add(current_theta, label*feature_vector)
    theta_0_update = current_theta_0 + label

    return (theta_update, theta_0_update)

def perceptron(feature_matrix, labels, T):
    """
    Section 1.4
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    if feature_matrix.ndim == 1:
        feature_matrix = np.array([feature_matrix])

    k = feature_matrix.shape[0]
    n = feature_matrix.shape[1]

    theta = np.array([0]*n)
    theta_0 = 0

    for j in xrange(T):
        for i in xrange(k):
            if labels[i] != np.sign((np.dot(theta,feature_matrix[i])+theta_0)):
                theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
    return (theta, theta_0)


def passive_aggressive_single_step_update(feature_vector, label, L, current_theta, current_theta_0):
    """
    Section 1.5
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the passive-aggressive algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the passive-aggressive
            algorithm parameters.
        current_theta - The current theta being used by the passive-aggressive
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            passive-aggressive algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    labels = np.array([label])

    x = hinge_loss(feature_vector, labels, current_theta, current_theta_0)/np.float((np.linalg.norm(feature_vector)**2+1))
    eta = min(x, 1/np.float(L))

    theta_update = np.add(current_theta, eta*label*feature_vector)
    theta_0_update = current_theta_0 + eta*label

    return (theta_update, theta_0_update)


def average_perceptron(feature_matrix, labels, T):
    """
    Section 1.6
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    k = feature_matrix.shape[0]
    n = feature_matrix.shape[1]

    theta = np.array([0]*n)
    theta_0 = 0

    theta_count = np.array([0]*n)
    theta_0_count = 0

    for j in xrange(T):

        for i in xrange(k):
            if labels[i]*(np.dot(theta,feature_matrix[i])+theta_0) <= 0:
                theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
                theta_count = np.add(theta_count, theta)
                theta_0_count = theta_0_count + theta_0
            else:
                theta_count = np.add(theta_count, theta)
                theta_0_count = theta_0_count + theta_0


    average_theta = theta_count / np.float(k*T)
    average_theta_0 = theta_0_count / np.float(k*T)
    return (average_theta, average_theta_0)

def average_passive_aggressive(feature_matrix, labels, T, L):
    """
    Section 1.6
    Runs the average passive-agressive algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the passive-agressive
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """

    k = feature_matrix.shape[0]
    n = feature_matrix.shape[1]

    #initialize the thetas to 0
    current_theta = np.array([0]*n)
    current_theta_0 = 0

    #running sum of the thetas
    theta_count = np.array([0]*n)
    theta_0_count = 0

    for j in xrange(T):
        for i in xrange(k):
            current_theta, current_theta_0 = passive_aggressive_single_step_update(feature_matrix[i], labels[i], L, current_theta, current_theta_0)
            theta_count = np.add(theta_count, current_theta)
            theta_0_count = np.add(theta_0_count, current_theta_0)


    avg_theta = theta_count / np.float(k*T)
    avg_theta_0 = theta_0_count / np.float(k*T)

    return (avg_theta, avg_theta_0)


### Part II

def classify(feature_matrix, theta, theta_0):
    """
    Section 2.8
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is the predicted
    classification of the kth row of the feature matrix using the given theta
    and theta_0.
    """

    if feature_matrix.ndim == 1:
        feature_matrix = np.array([feature_matrix])

    k = feature_matrix.shape[0]
    labels = []

    for i in xrange(k):
        if np.dot(feature_matrix[i],theta)+ theta_0 > 0:
            labels.append(1)
        else:
            labels.append(-1)

    labels = np.array(labels)
    return labels

def perceptron_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T):
    """
    Section 2.9
    Trains a linear classifier using the perceptron algorithm with a given T
    value. The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the perceptron algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """
    theta, theta_0 = perceptron(train_feature_matrix, train_labels, T)

    try_on_train = classify(train_feature_matrix, theta, theta_0)
    try_on_val = classify(val_feature_matrix, theta, theta_0)

    train_acc = accuracy(try_on_train, train_labels)
    val_acc = accuracy(try_on_val, val_labels)

    return (train_acc, val_acc)


def average_perceptron_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T):
    """
    Section 2.9
    Trains a linear classifier using the average perceptron algorithm with
    a given T value. The classifier is trained on the train data. The
    classifier's accuracy on the train and validation data is then returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the average perceptron
            algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """
    theta, theta_0 = average_perceptron(train_feature_matrix, train_labels, T)

    try_on_train = classify(train_feature_matrix, theta, theta_0)
    try_on_val = classify(val_feature_matrix, theta, theta_0)

    train_acc = accuracy(try_on_train, train_labels)
    val_acc = accuracy(try_on_val, val_labels)

    return (train_acc, val_acc)

def average_passive_aggressive_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T, L):
    """
    Section 2.9
    Trains a linear classifier using the average passive aggressive algorithm
    with given T and L values. The classifier is trained on the train data.
    The classifier's accuracy on the train and validation data is then
    returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the average passive
            aggressive algorithm.
        L - The value of L to use for training with the average passive
            aggressive algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """
    theta, theta_0 = average_passive_aggressive(train_feature_matrix, train_labels, T, L)

    try_on_train = classify(train_feature_matrix, theta, theta_0)
    try_on_val = classify(val_feature_matrix, theta, theta_0)

    train_acc = accuracy(try_on_train, train_labels)
    val_acc = accuracy(try_on_val, val_labels)

    return (train_acc, val_acc)

def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()

def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Section 3 (e.g. remove stopwords, add bigrams etc.)
    """
    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary:
                dictionary[word] = len(dictionary)

        # Uncomment below to add bigrams
        for i in xrange(len(word_list)-1):
            if word_list[i] + ' ' + word_list[i+1] not in dictionary:
                dictionary[word_list[i] + ' ' + word_list[i+1]] = len(dictionary)


    return dictionary

def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.
    """

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)

        for word in word_list:

            if word in dictionary:
                feature_matrix[i, dictionary[word]] = 1

        # Uncomment below to add bigrams.
        for j in xrange(len(word_list)-1):
            if word_list[j] + ' ' + word_list[j+1] in dictionary:
                feature_matrix[i, dictionary[word_list[j]+ ' ' + word_list[j+1]]] = 1

    return feature_matrix

def extract_additional_features(reviews):
    """
    Section 3.12
    Inputs a list of string reviews
    Returns a feature matrix of (n,m), where n is the number of reviews
    and m is the total number of additional features of your choice

    YOU MAY CHANGE THE PARAMETERS
    """

    num_reviews = len(reviews)
    feature_column = np.zeros([num_reviews,1])
    threshold = 3
    for i, text in enumerate(reviews):
        word_list = extract_words(text) #words in each review made into a list
        caps = 0
        for word in word_list:
            if word.isupper():
                caps = caps + 1
                if caps > threshold:
                    feature_column[i] = 1
                    break

    return feature_column

def extract_final_features(reviews, dictionary):
    """
    Section 3.12
    Constructs a final feature matrix using the improved bag-of-words and/or additional features
    """
    bow_feature_matrix = extract_bow_feature_vectors(reviews,dictionary)
    additional_feature_matrix = extract_additional_features(reviews)
    return np.hstack((bow_feature_matrix, additional_feature_matrix))

def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()
