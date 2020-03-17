import gzip
import pickle
from os import path

import openTSNE
from openTSNE import utils

with utils.Timer("Loading data...", verbose=True):
    with gzip.open(path.join("data", "macosko_2015.pkl.gz"), "rb") as f:
        data = pickle.load(f)

x = data["pca_50"]
y, cluster_ids = data["CellType1"], data["CellType2"]

# import sys; sys.path.append("FIt-SNE")
# from fast_tsne import fast_tsne
#
# with Timer("Running fast_tsne..."):
#     fast_tsne(x, nthreads=1)

affinities = openTSNE.affinity.PerplexityBasedNN(
    x,
    perplexity=30,
    metric="cosine",
    method="approx",
    n_jobs=-1,
    random_state=0,
    verbose=True,
)

init = openTSNE.initialization.spectral(affinities.P, verbose=True)

embedding = openTSNE.TSNEEmbedding(
    init,
    affinities,
    negative_gradient_method="fft",
    n_jobs=-1,
    random_state=0,
    verbose=True,
)

embedding.optimize(n_iter=250, exaggeration=12, momentum=0.5, inplace=True)
embedding.optimize(n_iter=500, momentum=0.8, inplace=True)


import matplotlib.pyplot as plt

plt.scatter(embedding[:, 0], embedding[:, 1], s=1)
plt.show()
