import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

class PCA:
    def __init__(self, n_components=None):

        self.n_components = n_components
        self.mean_ = None
        self.sorted_eig_val_ = None
        self.sorted_eig_vec_ = None
        
        # self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        
    def fit(self, X):
        # X: n*d matrix; n observations, d features
        
        n_samples, n_features = X.shape
        
        if self.n_components == None:
            self.n_components = min(n_samples, n_features)
            
        # mean vector
        self.mean_ = np.mean(X, axis = 0)
        
        # convariance matrix
        cov_mat = np.cov(X, rowvar=False)
        
        # eigenvalues and eigenvector
        eig_val, eig_vec = np.linalg.eig(cov_mat)
        eig_val = np.absolute(eig_val)
        
        # select top k eig_val
        idx = eig_val.argsort()[::-1]
        self.sorted_eig_val_ = eig_val[idx][0:self.n_components]
        self.sorted_eig_vec_ = eig_vec[:,idx][:,0:self.n_components]
        
        self.explained_variance_ratio_ = self.sorted_eig_val_/np.sum(eig_val)
        
        
    def transform(self, X):
        center_X = X - self.mean_
        #print center_X
        #print self.mean_
        #print self.sorted_eig_vec_
        trans_X = np.dot(center_X, self.sorted_eig_vec_) #+ self.mean_
        return trans_X
        


def test_function():

    np.random.seed(234) # random seed for consistency

    mu_vec1 = np.array([0,0,0])
    cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
    X = np.random.multivariate_normal(mu_vec1, cov_mat1, 20)
    assert X.shape == (20,3), "The matrix has not the dimensions 3x20"

    my_pca = PCA(n_components=2)
    my_pca.fit(X)
    print my_pca.transform(X)


if __name__ == "__main__":
    test_function()