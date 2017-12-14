import numpy as np
import scipy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class IterativeLDA:
    def __init__(self, n_components=1, verbose=False):
        self.n_components = n_components
        self.verbose=verbose
        
        self.ldas_ = []
        self.nullspaces_ = []
        
    def _ns_using_svd(self, A):
        u, s, v = np.linalg.svd(A)
        null_space = v[s.shape[0]:]
        return null_space.T
    
    def _ns_using_qr(self, A):
        q, r = scipy.linalg.qr(A.T)
        return q[:,1:]
    
    def _do_fit(self, X, y, n=None, new=True):
        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y should match on first dimension')
            
        if n is None:
            n = self.n_components
            
        if new:
            self.ldas_ = []
            self.nullspaces_ = []
        
        current_X = X
        transformed = []
        
        for i in range(n):
            if self.verbose: print('Fitting component {i}'.format(i=i + 1))
            lda = LinearDiscriminantAnalysis(n_components=1)
            transformed_X = lda.fit_transform(current_X, y)
            transformed.append(transformed_X)
            self.ldas_.append(lda)
            
            if self.verbose: print('Computing nullspace')
            ns = self._ns_using_qr(lda.coef_)
            if self.verbose: print('Nullspace computed')
            self.nullspaces_.append(ns)
            if self.verbose: print('Projecting onto nullspace')
            current_X = current_X.dot(ns)
        
        return transformed
        
    def fit(self, X, y):
        self._do_fit(X, y)
        return self
    
    def fit_transform(self, X, y):
        return np.hstack(self._do_fit(X, y))
        
    def transform(self, X, y=None):
        if 0 == len(self.ldas_) or 0 == len(self.nullspaces_):
            raise ValueError('Must call fit or fit_transform before calling transform')
        
        transformed = []
        current_X = X
        
        for lda, ns in zip(self.ldas_, self.nullspaces_):
            if self.verbose: print('Starting transform')
            transformed.append(lda.transform(current_X))
            if self.verbose: print('Computing nullspace')
            current_X = current_X.dot(ns)
            
        return np.hstack(transformed)    
    
    def add_components(self, n, X, y):
        current_X = X
        for ns in self.nullspaces_:
            current_X = current_X.dot(ns)
            
        self._do_fit(current_X, y, n, False)
        return self
        
    