import tensorflow as tf


class TfLogger:

    def __init__(self, log_dir):
        self.writer = tf.summary.FileWriter(log_dir)

    def log_all(self, summaries, step):
        for summary in summaries:
            self.log(summary, step)

    def log(self, summary, step):
        self.writer.add_summary(summary, step)

    def log_scalar(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.log(summary, step)

    def flush(self):
        self.writer.flush()
