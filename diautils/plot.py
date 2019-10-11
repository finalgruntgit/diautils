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


def plot_confusion_matrix(truth, pred, title='', xlabel='Prediction', ylabel='Truth'):
    cm = confusion_matrix(truth, pred)
    cm = np.around(cm.astype('float') / np.maximum(1, cm.sum(axis=1)[:, np.newaxis]), decimals=2)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    if title:
        plt.title(title)
    plt.colorbar()
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
