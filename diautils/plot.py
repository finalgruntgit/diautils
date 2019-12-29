import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib import cm
import numpy as np
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
    print(cm_normed)
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
