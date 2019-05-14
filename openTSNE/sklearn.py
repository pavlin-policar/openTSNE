import openTSNE
import numpy as np


class TSNE(openTSNE.TSNE):
    __doc__ = openTSNE.TSNE.__doc__

    def fit(self, X, y=None):
        """Fit X into an embedded space.

        Parameters
        ----------
        X: np.ndarray
            The data matrix to be embedded.
        y : ignored

        """
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        """Fit X into an embedded space and return that transformed output.

        Parameters
        ----------
        X: np.ndarray
            The data matrix to be embedded.
        y : ignored

        Returns
        -------
        np.ndarray
            Embedding of the training data in low-dimensional space.

        """
        embedding = super().fit(X)
        self.embedding_ = embedding
        return self.embedding_.view(np.ndarray)

    def transform(self, X, *args, **kwargs):
        """Apply dimensionality reduction to X.

        See :meth:`openTSNE.TSNEEmbedding.transform` for additional parameters.

        Parameters
        ----------
        X: np.ndarray
            The data matrix to be embedded.

        Returns
        -------
        np.ndarray
            Embedding of the training data in low-dimensional space.

        """
        embedding = self.embedding_.transform(X, *args, **kwargs)
        return embedding.view(np.ndarray)
