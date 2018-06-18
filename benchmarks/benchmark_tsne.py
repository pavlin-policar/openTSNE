import gzip
import os
import pickle
import time
import urllib

import fire
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE as SKLTSNE
from MulticoreTSNE import MulticoreTSNE

from Orange.data import Table
from Orange.projection import LDA, FreeViz
from tsne.tsne import TSNE, TSNEModel
import matplotlib.pyplot as plt


np.set_printoptions(precision=4, suppress=True)


def plot(x: np.ndarray, y: np.ndarray) -> None:
    for yi in np.unique(y):
        mask = y == yi
        plt.plot(x[mask, 0], x[mask, 1], 'o', label=str(yi), alpha=0.5, ms=3)
    plt.legend()


def plot1d(x: np.ndarray, y: np.ndarray) -> None:
    for yi in np.unique(y):
        mask = y == yi
        jitter = np.random.randn(mask.shape[0])
        plt.plot(x, jitter, 'o', label=str(yi), alpha=0.5, ms=3)
    plt.legend()


def get_mnist_full():
    if not os.path.exists('mnist.pkl.gz'):
        urllib.request.urlretrieve(
            'http://deeplearning.net/data/mnist/mnist.pkl.gz', 'mnist.pkl.gz')

    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train, val, test = pickle.load(f, encoding='latin1')
    _train = np.asarray(train[0], dtype=np.float64)
    _val = np.asarray(val[0], dtype=np.float64)
    _test = np.asarray(test[0], dtype=np.float64)
    mnist = np.vstack((_train, _val, _test))
    classes = np.hstack((train[1], val[1], test[1]))

    return mnist, classes


def check_transform():
    data = Table('iris')[::10]

    # lda_model = LDA()(data)
    # freeviz_model = FreeViz()(data)
    tsne_model = TSNE()(data)

    # lda_data = lda_model(data)
    # freeviz_data = freeviz_model(data)
    tsne_data = tsne_model(data)
    print(tsne_data)


def run():
    # mnist = datasets.load_digits()
    # x, y = mnist['data'], mnist['target']
    np.random.seed(1)
    x, y = get_mnist_full()
    indices = np.random.choice(list(range(x.shape[0])), 500, replace=False)
    x, y = x[indices], y[indices]

    # data = Table('cdp_expression_shekhar.pickle')
    # data = Table('sc-aml-sample.pickle')
    # data = Table('iris')
    # x, y = data.X, data.Y

    angle = 0.5
    perplexity = 20
    ee = 12
    lr = 1
    threads = 4
    metric = 'euclidean'

    print(x.shape)

    def plot_callback(error, embedding):
        plt.title('tsne')
        plot(embedding, y)
        plt.show()
        return True

    start = time.time()
    tsne = TSNE(
        perplexity=perplexity, learning_rate=lr, early_exaggeration=ee,
        n_jobs=threads, angle=angle, init='pca', metric=metric, n_components=2,
        n_iter=750, early_exaggeration_iter=250, neighbors='exact', grad='fft',
        late_exaggeration_iter=100, late_exaggeration=2., callback=plot_callback,
    )
    # x = PCA(n_components=50).fit_transform(x)
    embedding = tsne.fit(x)
    print('tsne', time.time() - start)
    plt.title('tsne')
    plot(embedding, y)
    plt.show()

    # pca_embedding = PCA(n_components=50).fit_transform(x)
    # embedding = tsne.fit(pca_embedding)
    # plt.title('tsne 50 pca')
    # plot(embedding, y)
    # plt.show()

    # init = PCA(n_components=2).fit_transform(x)
    # start = time.time()
    # embedding = MulticoreTSNE(
    #     early_exaggeration=ee, learning_rate=lr, perplexity=perplexity,
    #     n_jobs=threads, cheat_metric=False, angle=angle, init=init,
    #     metric=metric, verbose=True
    # ).fit_transform(x)
    # print('mctsne', time.time() - start)
    # plt.title('mctsne')
    # plot(embedding, y)
    # plt.show()

    # start = time.time()
    # embedding = SKLTSNE(
    #     early_exaggeration=ee, learning_rate=lr, angle=angle,
    #     perplexity=perplexity, init='pca', metric=metric,
    # ).fit_transform(x)
    # print('sklearn', time.time() - start)
    # plt.title('sklearn')
    # plot(embedding, y)
    # plt.show()


def transform():
    data = Table('cdp_expression_shekhar.pickle')
    data.shuffle()
    N = data.X.shape[0]
    # data = Table('sc-aml-sample.pickle')
    train, test = data[:N // 8], data[N // 8:]

    start = time.time()
    model = TSNE(
        n_components=2, perplexity=5, early_exaggeration=4, init='random',
        n_jobs=8,
        # late_exaggeration=1.1, late_exaggeration_iter=250,
    )(train)  # type: TSNEModel
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
