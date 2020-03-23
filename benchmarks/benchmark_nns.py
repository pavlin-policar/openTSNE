import gzip
import pickle
from os import path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from openTSNE import nearest_neighbors
from openTSNE import utils

with utils.Timer("Loading data...", verbose=True):
    with gzip.open(path.join("data", "macosko_2015.pkl.gz"), "rb") as f:
        data = pickle.load(f)

x = data["pca_50"]
y, cluster_ids = data["CellType1"], data["CellType2"]

results = []
n_reps = 5

for sample_size in range(1000, 8_001, 1000):
    print("Sample size:", sample_size)
    indices = np.random.choice(range(x.shape[0]), size=sample_size)
    sample = x[indices]

    for i in range(n_reps):
        start = time()
        nn = nearest_neighbors.BallTree(metric="euclidean", n_jobs=1)
        nn.build(sample, k=15)
        results.append(("Ball Tree (1 core)", sample_size, time() - start))

    for i in range(n_reps):
        start = time()
        nn = nearest_neighbors.Annoy(metric="euclidean", n_jobs=1)
        nn.build(sample, k=15)
        results.append(("Annoy (1 core)", sample_size, time() - start))

    for i in range(n_reps):
        start = time()
        nn = nearest_neighbors.BallTree(metric="euclidean", n_jobs=4)
        nn.build(sample, k=15)
        results.append(("Ball Tree (4 cores)", sample_size, time() - start))

    for i in range(n_reps):
        start = time()
        nn = nearest_neighbors.Annoy(metric="euclidean", n_jobs=4)
        nn.build(sample, k=15)
        results.append(("Annoy (4 cores)", sample_size, time() - start))

df = pd.DataFrame(results, columns=["method", "size", "time"])
df.to_csv("benchmark_nns.csv")
ax = sns.lineplot(data=df, x="size", y="time", hue="method")
ax.set_ylim(0, ax.get_ylim()[1])
ax.legend(bbox_to_anchor=(1.1, 1))
plt.show()
