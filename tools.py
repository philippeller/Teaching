import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot(X, y=None, rand=True):
    fig, ax = plt.subplots(5,5)
    
    if rand:
        idx = np.random.choice(len(X), 25, replace=False)
    else:
        idx = np.arange(25)
    
    for i in range(25):
        axis = ax[i//5, i%5]
        axis.axis('off')
        s = int(np.sqrt(X.shape[-1]))
        axis.imshow(X[idx[i]].reshape(s,s), cmap='gray_r')
        if y is not None:
            axis.set_title('y = %s'%y[idx[i]])
            
def pca_reduce(X, n=2, **kwargs):
    p = PCA(n_components=n, **kwargs)
    p.fit_transform(X)
    reduced_data = p.transform(X)
    return reduced_data