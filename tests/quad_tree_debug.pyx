from openTSNE.quad_tree cimport QuadTree, Node
import numpy as np


def print_tree(QuadTree tree):
    _print_tree(&tree.root)


cdef _print_tree(Node * node, name=None, indent=0):
    """Print the quad tree in a readable textual format."""
    if not node.num_points:
        return

    directions = {0: 'SW', 1: 'NW', 2: 'SE', 3: 'NE'}

    # Print the correct indentation
    print('\t' * indent + '%s: %s (%d) %s' % (
        'Root' if name is None else name,
        ['', '[+]'][not node.is_leaf],
        node.num_points,
        _str_point(<double [:node.n_dims]>node.center_of_mass),
    ))

    if not node.is_leaf:
        for sector in range(1 << node.n_dims):
            _print_tree(&node.children[sector], directions[sector], indent + 1)


def _str_point(double[:] point):
    return '(%s)' % ', '.join('%.4f' % point[i] for i in range(point.shape[0]))


def plot_tree(QuadTree tree, data):
    assert isinstance(data, np.ndarray), '`data` must be np.ndarray'
    if not data.dtype == np.float64:
        data = data.astype(np.float64)
    _plot_tree(&tree.root, data)


cdef _plot_tree(Node * root, double[:, :] data):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_xticks([]), ax.set_yticks([]), ax.axis('off')

    centers = []

    _add_patch(ax, root, centers)
    centers = list(zip(*centers))
    xs = [p[0] for p in data]
    ys = [p[1] for p in data]

    plt.scatter(xs, ys, s=20)
    # plt.scatter(centers[0], centers[1], edgecolors="r", facecolors="none", s=10, linewidths=1)

    plt.savefig("quadtree.png", dpi=80, rasterize=True, transparent=True, bbox_inches="tight")
    plt.show()


cdef _add_patch(ax, Node * node, centers):
    import matplotlib.patches as patches
    min_bounds = np.asarray(<double [:node.n_dims]>node.center) - node.length / 2
    ax.add_patch(patches.Rectangle(
        min_bounds, node.length, node.length, fill=False
    ))
    if not node.is_leaf:
        for i in range(1 << node.n_dims):
            _add_patch(ax, &node.children[i], centers)

    if node.num_points > 0:
        centers.append([node.center_of_mass[0], node.center_of_mass[1]])
