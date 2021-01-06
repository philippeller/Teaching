import numpy as np
from scipy import stats


def kmeans(X, n, n_iter=100, centroids=None):
    """
    simple k-means implemenmtation
    
    Parameters:
    -----------
    X: array (n_samples, n_dim)
        the data
    n : int
        number of clusters
    n_iter : int
        number of iterations
    centroids : array (optional)
        
    Returns:
    --------
    
    centroids : array
    labels : array
    """

    def assign(centroids):
        dist = np.average(np.square(X[:, :, np.newaxis] - centroids.T), axis=1)        
        return np.argmin(dist, axis=1)        
    
    def get_centroids(labels):
        centroids = []
        for i in range(np.max(labels)+1):
            centroids.append(np.average(X[labels==i], axis=0))
        return np.stack(centroids)    
        
    if centroids is None:
        # start with random seed
        centroids = np.random.rand(n, X.shape[1])
        # scale to roughly match the data
        centroids *= np.std(X, axis=0)
        centroids += np.mean(X, axis=0)
            
    # iterate
    for i in range(n_iter):
        labels = assign(centroids)
        centroids = get_centroids(labels)

    return centroids, labels


def gmm(X, n, n_iter=100, mus=None, covs=None):
    """
    simple GMM implemenmtation (2d right now)
    
    Parameters:
    -----------
    X: array (n_samples, n_dim)
        the data
    n : int
        number of gaussians
    n_iter : int
        number of iterations
    mus : array (optional)
    covs = array (optional)
        
    Returns:
    --------
    mus : array
        mean values
    covs : array
        covariances
    exps : array
        probabilities
    """
    
    
    def E(mus, covs):
        
        probs = []
        exps = []
        
        for i in range(n):    
            probs.append(stats.multivariate_normal(mus[i], covs[i]).pdf(X))
    
        probs = np.array(probs)
        tot_prob = np.sum(probs, axis=0)
        
        for i in range(n):
            exps.append(probs[i] / tot_prob)
        
        return np.nan_to_num(exps)
    
    def M(exps):
        
        mus = []
        covs = []
        
        for i in range(n): 
            mus.append(np.average(X, weights=exps[i], axis=0))
            covs.append(np.cov(X.T, aweights=exps[i]))
        
        return mus, covs
    
    
    if mus is None:
        std = np.std(X, axis=0)
        mus = np.random.rand(n, X.shape[1])
        # scale to roughly match the data
        mus *= std
        mus += np.mean(X, axis=0)
        # some random covariances
        covs = np.random.rand(n, X.shape[1], X.shape[1])
        covs = covs @ np.swapaxes(covs, 1, 2)
    
    
    for i in range(n_iter):
        exps = E(mus, covs)
        mus, covs = M(exps)
        
    return mus, covs, np.array(E(mus, covs)).T
    