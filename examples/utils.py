from os.path import abspath, dirname, join
import pickle
import numpy as np


FILE_DIR = dirname(abspath(__file__))
DATA_DIR = join(FILE_DIR, 'data')


def get_zeisel_2018(n_samples: int = None):
    with open(join(DATA_DIR, 'zeisel_2018.pickle'), 'rb') as f:
        data = pickle.load(f)
    
    # Extract log normalized counts and cell type
    x, y = data['log_counts'], data['CellType1']
    x = x.T

    if n_samples is not None:
        indices = np.random.choice(list(range(x.shape[0])), n_samples, replace=False)
        x, y = x[indices], y[indices]

    return x, y
    
    
def plot(x: np.ndarray, y: np.ndarray, ax=None, draw_legend=True, **kwargs) -> None:
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
    
    for yi in np.unique(y):
        mask = y == yi
        ax.plot(x[mask, 0], x[mask, 1], 'o', label=str(yi),
                alpha=kwargs.get('alpha', 0.6), ms=kwargs.get('ms', 1))
    
    # Hide ticks
    ax.set_xticks([]), ax.set_yticks([])
    # Hide box border around figure
    ax.axis('off')
    
    if draw_legend:
        legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # Make the legend markers bigger, so we can see what's what
        for lh in legend.legendHandles:
            lh._legmarker.set_markersize(6)
            lh._legmarker.set_alpha(1)
