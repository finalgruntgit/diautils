import matplotlib.pyplot as plt


def plot_fs():
    backend = plt.get_backend()
    print(backend)
    if backend == 'Qt4Agg' or backend == 'Qt5Agg':
        plt.get_current_fig_manager().window.showMaximized()
    elif backend == 'TkAgg':
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
    plt.show()
