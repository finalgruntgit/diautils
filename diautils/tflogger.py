import tensorflow as tf
from diautils.plot import *


class TfLogger:

    def __init__(self, log_dir):
        self.writer = tf.summary.FileWriter(log_dir)

    def log_all(self, summaries, step):
        for summary in summaries:
            self.log(summary, step)
        return self

    def log(self, summary, step):
        self.writer.add_summary(summary, step)
        return self

    def log_hist(self, tag, values, step, bins=1000):
        """
        Logs the histogram of a list/vector of values.
        From: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
        """

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Therefore we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        return self.log(summary, step)

    def log_scalar(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        return self.log(summary, step)

    def log_image_bytes(self, tag, value, step):
        img_sum = tf.Summary.Image(encoded_image_string=value)
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, image=img_sum)])
        return self.log(summary, step)

    def log_plot(self, tag, step):
        return self.log_image_bytes(tag, plot_to_img_bytes().getvalue(), step)

    def log_cm(self, tag, value, step, title='', xlabel='Prediction', ylabel='Truth'):
        plot_confusion_matrix(value, title, xlabel, ylabel)
        return self.log_plot(tag, step)

    def flush(self):
        self.writer.flush()
        return self