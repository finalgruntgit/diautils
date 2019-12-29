import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib import cm
import numpy as np
import networkx as nx
import io


def plot_fs():
    backend = plt.get_backend()
    print(backend)
    if backend == 'Qt4Agg' or backend == 'Qt5Agg':
        plt.get_current_fig_manager().window.showMaximized()
    elif backend == 'TkAgg':
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
    plt.show()


def plot_save(filename, close=True):
    plt.savefig(filename)
    if close:
        plt.close()


def plot_confusion_matrix(truth, pred, classes=None, title='', xlabel='Prediction', ylabel='Truth', cmap_name='Blues', show_colorbar=False, relative=False, show_ratio=True, ratio_precision=2, threshold=0.5):
    cm = confusion_matrix(truth, pred)
    cm_normed = cm.astype(np.float32) / cm.sum(axis=1)[:, None]
    if classes is None:
        classes = np.arange(max(cm.shape[0], cm.shape[1]))
    if relative:
        plt.imshow(cm_normed, interpolation='nearest', cmap=plt.get_cmap(cmap_name))
    else:
        plt.imshow(cm_normed, interpolation='nearest', cmap=plt.get_cmap(cmap_name), vmin=0.0, vmax=1.0)
    if title:
        plt.title(title)
    if show_colorbar:
        plt.colorbar()
    if show_ratio:
        fmt = '{{}}\n({{:.{}f}})'.format(ratio_precision)
    else:
        fmt = '{}'
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, fmt.format(cm[i, j], cm_normed[i, j]), color='white' if cm_normed[i, j] >= threshold else 'black', horizontalalignment='center', verticalalignment='center')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)


def plot_to_img_bytes():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf


def create_colormap(name='tab20', num_color=12):
    cmap = cm.get_cmap(name, num_color)
    return cmap(np.linspace(0, 1, num_color))


class GraphLayout:

    def __init__(self, g, pos, nodes, edges, node_labels=None, node_cmap='Greens', node_size_min=10, node_size_max=140, node_linewidth=1, edge_cmap='Greens', edge_linewidth=2, show_arrows=False):
        self.g = g
        self.pos = pos
        self.nodes = nodes
        self.edges = edges
        self.node_labels = node_labels
        self.node_cmap = plt.get_cmap(node_cmap)
        self.node_size_min = node_size_min
        self.node_size_max = node_size_max
        self.node_linewidth = node_linewidth
        self.edge_cmap = plt.get_cmap(edge_cmap)
        self.edge_linewidth = edge_linewidth
        self.show_arrows = show_arrows
        self.node_labels = None if node_labels is None else dict((v, node_labels[i]) for i, v in enumerate(self.nodes))

    def plot(self, node_weights=None, edge_weights=None):
        if node_weights is None:
            nx.draw_networkx_nodes(self.g, self.pos, nodelist=self.nodes, cmap=self.node_cmap, vmin=0.0, vmax=1.0, node_color=np.repeat(0.5, len(self.nodes)), node_size=0.5 * (self.node_size_min + self.node_size_max), linewidths=self.node_linewidth, edgecolors='black')
        else:
            nx.draw_networkx_nodes(self.g, self.pos, nodelist=self.nodes, cmap=self.node_cmap, vmin=0.0, vmax=1.0, node_color=node_weights, node_size=self.node_size_min + self.node_size_max * node_weights, linewidths=1, edgecolors='black')
        nx.draw_networkx_edges(self.g, self.pos, edgelist=self.edges, edge_color='black', arrows=False, width=self.edge_linewidth + 0.5)
        if edge_weights is None:
            nx.draw_networkx_edges(self.g, self.pos, edgelist=self.edges, edge_cmap=self.edge_cmap, edge_vmin=0.0, edge_vmax=1.0, edge_color=np.repeat(0.5, len(self.edges)), arrows=self.show_arrows, width=self.edge_linewidth)
        else:
            nx.draw_networkx_edges(self.g, self.pos, edgelist=self.edges, edge_cmap=self.edge_cmap, edge_vmin=0.0, edge_vmax=1.0, edge_color=edge_weights, arrows=self.show_arrows, width=self.edge_linewidth)
        if self.node_labels is not None:
            nx.draw_networkx_labels(self.g, self.pos, labels=self.node_labels)
        return self


def create_graph_layout(nodes, edges, directed=True, type='spring', iterations=100, node_labels=None, node_cmap='Blues', node_size_min=10, node_size_max=140, node_linewidth=1, edge_cmap='Greens', edge_linewidth=2, show_arrows=False):
    if directed:
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    if isinstance(nodes, np.ndarray):
        nodes = nodes.tolist()
    elif isinstance(nodes, int):
        nodes = list(range(nodes))
    for v in nodes:
        g.add_node(v)
    for edge in edges:
        g.add_edge(*edge)
    if type == 'spring':
        pos = nx.spring_layout(g, iterations=iterations)
    elif type == 'kawai':
        pos = nx.kamada_kawai_layout(g)
    else:
        raise Exception('Unknown graph layout type: {}'.format(type))
    return GraphLayout(g, pos, nodes, edges, node_labels, node_cmap, node_size_min, node_size_max, node_linewidth, edge_cmap, edge_linewidth, show_arrows)
