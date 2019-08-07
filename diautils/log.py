import enum


class LogLevel(enum.IntEnum):
    FATAL = 0,
    ERROR = 1,
    WARN = 2,
    INFO = 3,
    DEBUG = 4,
    ALL = 5,


class Logger:

    def __init__(self, level=LogLevel.INFO):
        self.level = level

    def set(self, level=LogLevel.INFO):
        self.level = level
        return self

    def log(self, level, *args, sep=' ', end='\n', file=None):
        if level <= self.level:
            print(*args, sep=sep, end=end, file=file)
        return self

    def all(self, *args, sep=' ', end='\n', file=None):
        return self.log(LogLevel.ALL, *args, sep=sep, end=end, file=file)

    def debug(self, *args, sep=' ', end='\n', file=None):
        return self.log(LogLevel.DEBUG, *args, sep=sep, end=end, file=file)

    def info(self, *args, sep=' ', end='\n', file=None):
        return self.log(LogLevel.INFO, *args, sep=sep, end=end, file=file)

    def warn(self, *args, sep=' ', end='\n', file=None):
        return self.log(LogLevel.WARN, *args, sep=sep, end=end, file=file)

    def error(self, *args, sep=' ', end='\n', file=None):
        return self.log(LogLevel.ERROR, *args, sep=sep, end=end, file=file)

    def fatal(self, *args, sep=' ', end='\n', file=None):
        return self.log(LogLevel.FATAL, *args, sep=sep, end=end, file=file)


log = Logger()
