# A SVM classifier
import numpy as np
import pandas as pd
from cvxopt import matrix
from cvxopt.solvers import qp
from cvxopt.solvers import options
from scipy.spatial.distance import pdist, squareform
from scipy import exp


class SVM():
    def __init__(self, gamma, C=100, kernel='rbf'):
        
        self.C = C # Penalty parameter C
        self.kernel = kernel 
        # self.degree = degree # poly kernel is not implemented
        self.gamma = gamma # Kernel coefficient for rbf
        self.c = None # see wikipedia SVM, dual form
        self.w = None # the coefficient for the separating plane
    
    @staticmethod
    def linear_kernel(X):
        return np.dot(X, X.T)
    
    @staticmethod
    def rbf_kernel(X, gamma):
        pairwise_dists = squareform(pdist(X, 'euclidean'))
        K = exp(-pairwise_dists ** 2 * gamma)
        return K
        
    def fit(self, X, y):
        options['show_progress'] = 0
        
        # X: n*d matrix
        # y: n*1 numpy array
        n = X.shape[0]
        
        # XXT = np.dot(X, X.T)
        if self.kernel == 'linear':
            XXT = self.linear_kernel(X)
        elif self.kernel == 'rbf':
            XXT = self.rbf_kernel(X, self.gamma)
        else:
            raise Exception('Kernel function is not implemented.')
            
        P = matrix(np.outer(y, y) * XXT, tc = 'd')
        q = matrix(-1.0, (n, 1))
        
        A = matrix(y, (1, n), tc = 'd')
        b = matrix(0.0)

        G = matrix(np.concatenate([np.eye(n) * -1.0, np.eye(n)]), tc = 'd')
        h = matrix(np.concatenate([np.zeros(n), np.ones(n) * self.C]))
                              
        # Solving the QP problem
        solution = qp(P, q, G, h, A, b)
        self.c = np.ravel(solution['x']) 
        #self.c[self.c < 1e-4] = 0
        
        # get the coefficint w
        self.w = np.dot(self.c*y, X)
        
        #self.c[self.c > (1.0/(2*n*self.C) - 1e-3)] = 0
        temp_b = y - np.dot(X, self.w)
        
        self.b = temp_b[(self.c > 1e-4) & (self.c < (self.C - 1e-4))]
        # print self.b
        if self.b.shape[0] == 0:
            print self.c.tolist()
            # self.b = 0
            raise Exception('No offset value is found!')
        else:
            self.b = self.b[0]
        
    def predict(self, X):
        
        label = np.dot(X, self.w) + self.b
        
        label[label>=0] = 1
        label[label<0] = -1
        return label


def test_function():
    data_dict = np.array([[1,7, -1],
                         [2,8, -1],
                         [3,8, -1],
             
                         [5,1, 1],
                         [6,-1,1],
                         [7,3, 1 ]])

    data_dict = data_dict *1.0

    # C = 100 works
    my_svm = SVM(gamma = 0.0001)
    my_svm.fit(data_dict[:,0:2],data_dict[:,2])
    predicted = my_svm.predict(data_dict[:,0:2])

    print predicted

if __name__ == "__main__":
    test_function()


