import matplotlib.pyplot as plt
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


def plot_confusion_matrix(cm):
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('matrice de confusion')
    plt.colorbar()

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Prediction Label')
    return figure


def plot_to_img_bytes(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    return buf


def create_colormap(name='tab20', num_color=12):
    cmap = cm.get_cmap(name, num_color)
    return cmap(np.linspace(0, 1, num_color))
