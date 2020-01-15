from diautils.plot import *
from torch.utils.tensorboard import SummaryWriter

class TorchLogger:

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def log_hist(self, tag, values, step, bins='auto'):
        self.writer.add_histogram(tag, values, step, bins)
        return self

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
        return self

    def log_image(self, tag, value, step):
        self.writer.add_image(tag, value, step, dataformats='HWC')
        return self

    def log_plot(self, tag, step):
        return self.log_image(tag, plot_to_img_np(), step)

    def log_parameters(self, tag, value, step, normed=True):
        plot_parameters(value, normed)
        return self.log_plot(tag, step)

    def log_cm(self, tag, truth, pred, step, classes=None, title='', xlabel='Prediction', ylabel='Truth', cmap_name='Blues', show_colorbar=False, relative=False, show_ratio=True, ratio_precision=2, threshold=0.5):
        plot_confusion_matrix(truth, pred, classes, title, xlabel, ylabel, cmap_name, show_colorbar, relative, show_ratio, ratio_precision, threshold)
        return self.log_plot(tag, step)

    def log_graph(self, tag, layout, step, node_weights=None, edge_weights=None):
        layout.plot(node_weights, edge_weights)
        return self.log_plot(tag, step)

    def flush(self):
        self.writer.flush()
        return self