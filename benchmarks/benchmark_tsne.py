import gzip
import os
import pickle
import time
import urllib
from os.path import abspath, dirname, join

import fire
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE as SKLTSNE
from MulticoreTSNE import MulticoreTSNE

from tsne.tsne import TSNE, TSNEEmbedding
import matplotlib.pyplot as plt


FILE_DIR = dirname(abspath(__file__))
DATA_DIR = join(FILE_DIR, 'data')


np.set_printoptions(precision=4, suppress=True)


def plot(x: np.ndarray, y: np.ndarray) -> None:
    for yi in np.unique(y):
        mask = y == yi
        plt.plot(x[mask, 0], x[mask, 1], 'o', label=str(yi), alpha=0.5, ms=1)
    plt.legend()
    plt.show()


def plot1d(x: np.ndarray, y: np.ndarray) -> None:
    for yi in np.unique(y):
        mask = y == yi
        jitter = np.random.randn(mask.shape[0])
        plt.plot(x, jitter, 'o', label=str(yi), alpha=0.5, ms=1)
    plt.legend()
    plt.show()


def get_mnist_full():
    if not os.path.exists('data/mnist.pkl.gz'):
        urllib.request.urlretrieve(
            'http://deeplearning.net/data/mnist/mnist.pkl.gz', 'data/mnist.pkl.gz')

    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        train, val, test = pickle.load(f, encoding='latin1')
    _train = np.asarray(train[0], dtype=np.float64)
    _val = np.asarray(val[0], dtype=np.float64)
    _test = np.asarray(test[0], dtype=np.float64)
    mnist = np.vstack((_train, _val, _test))
    classes = np.hstack((train[1], val[1], test[1]))

    return mnist, classes


def tmp():
    from tsne.tsne import TSNE

    iris = datasets.load_iris()
    x = iris['data']
    y = iris['target']

    tsne = TSNE(
        perplexity=30, learning_rate=100, early_exaggeration=12,
        n_jobs=4, angle=0.5, initialization='pca', metric='euclidean',
        n_components=2, n_iter=750, early_exaggeration_iter=250, neighbors='exact',
        negative_gradient_method='bh', min_num_intervals=10, ints_in_inverval=2,
        late_exaggeration_iter=0, late_exaggeration=4,
    )
    embedding = tsne.get_initial_embedding_for(x)
    embedding = embedding.optimize(n_iter=250, exaggeration=12, momentum=0.5)
    embedding = embedding.optimize(n_iter=750, momentum=0.8)
    plot(embedding, y)

    embedding = tsne.fit(x)
    plot(embedding, y)


def run():
    # mnist = datasets.load_digits()
    # x, y = mnist['data'], mnist['target']
    # np.random.seed(1)
    # x, y = get_mnist_full()
    # indices = np.random.choice(list(range(x.shape[0])), 5000, replace=False)
    # x, y = x[indices], y[indices]

    with open(join(DATA_DIR, 'sc-mouse-60k-1k.pkl'), 'rb') as f:
        x = pickle.load(f)
    x = x.astype(np.float32).toarray()
    import json
    with open(join(DATA_DIR, 'sc-mouse-60k-1k-y.txt'), 'rb') as f:
        y = np.array(json.load(f), dtype=int)

    angle = 0.5
    perplexity = 30
    ee = 12
    lr = 100
    threads = 4
    metric = 'euclidean'

    print(x.shape)

    start = time.time()
    tsne = TSNE(
        perplexity=perplexity, learning_rate=lr, early_exaggeration=ee,
        n_jobs=threads, angle=angle, initialization='random', metric=metric,
        n_components=2, n_iter=750, early_exaggeration_iter=250, neighbors='approx',
        negative_gradient_method='fft', min_num_intervals=10, ints_in_inverval=2,
        late_exaggeration_iter=0, late_exaggeration=2.,
    )
    # x = PCA(n_components=50).fit_transform(x)
    embedding = tsne.fit(x)
    print('-' * 80)
    print('tsne', time.time() - start)
    plt.title('tsne')
    plot(embedding, y)

    init = PCA(n_components=2).fit_transform(x)
    start = time.time()
    embedding = MulticoreTSNE(
        early_exaggeration=ee, learning_rate=lr, perplexity=perplexity,
        n_jobs=threads, cheat_metric=False, angle=angle, init=init,
        metric=metric, verbose=True
    ).fit_transform(x)
    print('-' * 80)
    print('mctsne', time.time() - start)
    plt.title('mctsne')
    plot(embedding, y)
    plt.show()

    x = np.ascontiguousarray(x.astype(np.float64))
    from fitsne import FItSNE
    start = time.time()
    embedding = FItSNE(
        x, 2, perplexity=perplexity, stop_lying_iter=250, ann_not_vptree=True,
        early_exag_coeff=ee, nthreads=threads, theta=angle,
    )
    print('-' * 80)
    print('fft interp %.4f' % (time.time() - start))
    plt.title('fft interp')
    plot(embedding, y)
    plt.show()

    start = time.time()
    embedding = SKLTSNE(
        early_exaggeration=ee, learning_rate=lr, angle=angle,
        perplexity=perplexity, init='pca', metric=metric,
    ).fit_transform(x)
    print('-' * 80)
    print('sklearn', time.time() - start)
    plt.title('sklearn')
    plot(embedding, y)
    plt.show()


def transform():
    data = Table('cdp_expression_shekhar.pickle')
    data.shuffle()
    N = data.X.shape[0]
    # data = Table('sc-aml-sample.pickle')
    train, test = data[:N // 8], data[N // 8:]

    start = time.time()
    model = TSNE(
        n_components=2, perplexity=5, early_exaggeration=4, initialization='random',
        n_jobs=8,
        # late_exaggeration=1.1, late_exaggeration_iter=250,
    )(train)  # type: TSNEEmbedding
    print('tsne train', time.time() - start)
    plot(model.embedding, train.Y)
    plt.gca().set_color_cycle(None)

    start = time.time()
    transformed = model(test, perplexity=10, exaggeration_iter=200)
    print('tsne trasnsform', time.time() - start)
    plot(transformed, test.Y)

    plt.show()


if __name__ == '__main__':
    fire.Fire()
