import pyximport; pyximport.install()
import quad_tree
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA

# Fetch MNIST
mnist = fetch_mldata('MNIST original')
X = mnist.data / 255
y = mnist.target

# Compute the PCA preprocessing projection
n_pca_components = 4
pca_columns = ['PC%d' % (i + 1) for i in range(n_pca_components)]
pca = PCA(n_components=n_pca_components)
pca_projection = pca.fit_transform(X)
print('PCA explained variance %.4f' % np.sum(pca.explained_variance_ratio_))
