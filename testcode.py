import numpy as np
import project1 as p1
# have tested:
# hinge loss
# perceptron
# average perceptron

# test_feature_matrix = np.array([[1,-1],[1,1]])
# test_theta = np.array([1,1])
# test_labels = np.array([1,-1])
# test_theta_0 = 0

#print p1.classify(test_feature_matrix ,test_theta, test_theta_0)



test_feature_matrix = np.array([[1,0,0],[0,1,1],[1,0,1],[0,1,0]])
test_labels = np.array([-1,1,-1,1])
test_theta = np.array([1,1,1])
test_theta_0 = 1
test_feature_vector = test_feature_matrix[0,:]
#print test_feature_vector
print 'testing hinge loss',  p1.hinge_loss(test_feature_matrix, test_labels, test_theta, test_theta_0)
#print p1.hinge_loss(test_feature_vector, test_labels, test_theta, test_theta_0)
#print 'testing perceptron', p1.perceptron(test_feature_matrix, test_labels, 1)
#print p1.average_perceptron(test_feature_matrix, test_labels, 3)
