from _ctypes import PyObj_FromPtr
import bz2
import collections as cols
import datetime
import enum
import fcntl
import gzip
import inspect
import json
import multiprocessing as mp
import numba as na
import numpy as np
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import threading
import traceback
import zipfile


# Collections
def merge_write(left_into, right_from):
    for k, v in right_from.items():
        if k in left_into and isinstance(left_into[k], dict) and isinstance(v, cols.Mapping):
            merge_write(left_into[k], v)
        else:
            left_into[k] = v
    return left_into


def merge(*dics, out=None):
    if out is None:
        out = {}
    for dic in dics:
        merge_write(out, dic)
    return out


def put_if_absent(dict, key, value):
    if key not in dict:
        dict[key] = value
    return dict


def ensure(value, default):
    return default if value is None else value


def recursive_flatten_iter(ls):
    for v in ls:
        if isinstance(v, cols.Iterable) and not isinstance(v, (str, bytes)):
            yield from recursive_flatten_iter(v)
        else:
            yield v


def recursive_flatten(ls):
    return list(recursive_flatten_iter(ls))


def flatten(l):
    for v in l:
        if isinstance(v, cols.Iterable) and not isinstance(v, (str, bytes)):
            yield from v
        else:
            yield v


def stack_iter(ls):
    for l in ls:
        yield list(flatten(l))


def stack(ls):
    return list(stack_iter(ls))


def repeat(value, num):
    for _ in range(num):
        yield value


def repeatable(value, num):
    class _Inner():

        def __iter__(self):
            return repeat(value, num)

    return _Inner()


def is_arraylike(obj):
    return hasattr(obj, '__getitem__')


def is_iterable(obj):
    return hasattr(obj, '__iter__')


def is_iterator(obj):
    return hasattr(obj, '__next__')


def is_string(obj):
    return isinstance(obj, str)


@na.njit
def chunks_numba(l, step, end):
    for i in range(0, end, step):
        yield l[i:i + step]


def chunks(l, step, num=None):
    return chunks_numba(l, step, len(l) if num is None else num * step)


# Filesystem
def env(key):
    return os.environ[key]


def cwd():
    return os.getcwd()


def current_file_path():
    return inspect.stack()[1].filename


def cp_current_file(dest, new_name=None):
    if new_name is not None:
        dest = join(dest, new_name)
    cpfile(inspect.stack()[1].filename, dest)


def join(*paths):
    paths = [str(v) for v in paths if v is not None and len(str(v)) > 0]
    return os.path.join(paths[0], *paths[1:])


def listdir(path):
    return os.listdir(path)


def isfile(filename):
    return os.path.isfile(filename)


def isdir(filename):
    return os.path.isdir(filename)


def islink(filename):
    return os.path.islink(filename)


def basename(filename):
    return os.path.basename(filename)


def cpfile(from_path, to_path):
    try:
        shutil.copy(from_path, to_path)
    except:
        pass


def mvfile(from_path, to_path):
    try:
        shutil.move(from_path, to_path)
    except:
        pass


def rmfile(filename):
    try:
        os.remove(filename)
    except:
        pass


def rmdirs(dir):
    try:
        shutil.rmtree(dir, ignore_errors=True)
    except:
        pass


def mkdirs(dir):
    try:
        os.makedirs(dir)
    except:
        pass


def rmmkdirs(dir):
    rmdirs(dir)
    mkdirs(dir)


def cpdirs(from_path, to_path):
    try:
        shutil.copytree(from_path, to_path)
    except:
        pass


def symlink(src, dst, add_cwd=True):
    try:
        if add_cwd:
            cwd_ = cwd()
            src = join(cwd_, src)
            dst = join(cwd_, dst)
        os.symlink(src, dst)
    except:
        pass


def latest_file(dir):
    try:
        return sorted(listdir(dir))[-1]
    except:
        return None


def exists_file(path):
    return os.path.exists(path)


def file_cmd(filename):
    return subprocess.check_output(['file', filename])


def load_txt(filename, zip=False):
    if zip:
        filename = '{}.gz'.format(filename)
    if not exists_file(filename):
        return None
    if zip:
        with gzip.open(filename, 'rt') as file:
            return file.read()
    else:
        with open(filename, 'r') as file:
            return file.read()


def save_txt(filename, data, zip=False):
    if zip:
        with gzip.open('{}.gz'.format(filename), mode='wt') as file:
            file.write(data)
            return data
    else:
        with open(filename, mode='w') as file:
            file.write(data)
            return data


def md5_to_path(md5, depth=4, step=2):
    return join(*chunks(md5, step, depth))


def md5sum(file):
    try:
        return subprocess.check_output(['md5sum', file])[:32].decode('utf-8')
    except:
        return None


def sha1sum(file):
    try:
        return subprocess.check_output(['sha1sum', file])[:40].decode('utf-8')
    except:
        return None


def sha256sum(file):
    try:
        return subprocess.check_output(['sha256sum', file])[:64].decode('utf-8')
    except:
        return None


def gzip_file(file):
    if exists_file(file):
        try:
            subprocess.call(['gzip', file])
            return True
        except:
            pass
    return False


def gunzip_file(file):
    if exists_file(file):
        try:
            subprocess.call(['gunzip', file])
            return True
        except:
            pass
    return False


def gunzip_file_to(file_src, file_dest, block_size=65536):
    with gzip.open(file_src, 'rb') as fs, open(file_dest, 'wb') as fd:
        while True:
            block = fs.read(block_size)
            if not block:
                break
            else:
                fd.write(block)


def open_dir(dir_path):
    try:
        os.system('xdg-open "{}"'.format(dir_path))
    except:
        pass


def file_size(file):
    return os.path.getsize(file)


def untargz_file_to(file_src, file_dest):
    if exists_file(file_src):
        with tarfile.open(file_src, 'r:gz') as tar:
            tar.extractall(file_dest)


def untarbz2_file_to(file_src, file_dest):
    if exists_file(file_src):
        with tarfile.open(file_src, 'r:bz2') as tar:
            tar.extractall(file_dest)


def untar_file_to(file_src, file_dest):
    if exists_file(file_src):
        with tarfile.open(file_src, 'r:') as tar:
            tar.extractall(file_dest)


def unzip_file_to(file_src, file_dest):
    if exists_file(file_src):
        with zipfile.ZipFile(file_src, 'r') as zfile:
            zfile.extractall(file_dest)


def bz2unzip_file_to(file_src, file_dest, block_size=65536):
    with bz2.BZ2File(file_src, 'r') as fs, open(file_dest, 'wb') as fd:
        while True:
            block = fs.read(block_size)
            if not block:
                break
            else:
                fd.write(block)


# Sync
class SystemLock():

    def __init__(self, filename='.lock', mode='w+'):
        self.filename = filename
        self.mode = mode
        self.lock = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.release()

    def acquire(self):
        if self.lock is None:
            self.lock = open(self.filename, self.mode)
            fcntl.lockf(self.lock, fcntl.LOCK_EX)

    def release(self):
        if self.lock is not None:
            fcntl.lockf(self.lock, fcntl.LOCK_UN)
            self.lock = None


# Time
def current_time():
    return time.time()


def current_time_millis():
    return int(current_time() * 1000)


def current_time_str(format="%Y-%m-%d_%Hh%Mm%Ss"):
    return datetime.datetime.now().strftime(format)


class Timer():

    def __init__(self):
        self.t = 0

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self.last = current_time()
        return self

    def stop(self):
        self.t += current_time() - self.last
        return self

    def update(self):
        now = current_time()
        self.t += now - self.last
        self.last = now
        return self

    def reset(self):
        self.t = 0
        return self

    def restart(self):
        return self.reset().start()

    def time(self):
        return self.t

    def time_millis(self):
        return int(self.t * 1000)

    def time_sec(self):
        return int(self.t)


# Json
class NoIndent(object):
    def __init__(self, value):
        self.value = value


class DiaJsonEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

    def __init__(self, **kwargs):
        self.__sort_keys = kwargs.get('sort_keys', None)
        super().__init__(**kwargs)

    def default(self, obj):
        if isinstance(obj, NoIndent):
            return self.FORMAT_SPEC.format(id(obj))
        else:
            return super().default(obj)

    def encode(self, obj):
        format_spec = self.FORMAT_SPEC
        json_repr = super().encode(obj)
        for match in self.regex.finditer(json_repr):
            id = int(match.group(1))
            no_ident = PyObj_FromPtr(id)
            json_obj_repr = json.dumps(no_ident.value, sort_keys=self.__sort_keys)
            json_repr = json_repr.replace('"{}"'.format(format_spec.format(id)), json_obj_repr)
        return json_repr


def format_json_pretty(obj, indent=4, cls=DiaJsonEncoder):
    return json.dumps(obj, indent=indent, cls=cls)


def save_json_pretty(file, data=None, indent=4, cls=DiaJsonEncoder, zip=False):
    return save_json(file, data, indent, cls, zip)


def save_json(filename, data=None, indent=None, cls=DiaJsonEncoder, zip=False):
    return save_txt(filename, format_json_pretty({} if data is None else data, indent, cls), zip)


def load_json(filename, zip=False):
    if zip:
        filename = '{}.gz'.format(filename)
    if not exists_file(filename):
        return None
    if zip:
        with gzip.open(filename, 'rt') as file:
            return json.load(file)
    else:
        with open(filename, 'r') as file:
            return json.load(file)


def parse_json(data):
    return json.loads(data)


def json_skip(value):
    return NoIndent(value)


# Numpy
def save_npy(file, data, allow_pickle=False, fix_imports=False):
    np.save(file, data, allow_pickle=allow_pickle, fix_imports=fix_imports)


def load_npy(file, allow_pickle=False, fix_imports=False):
    file = npyfile(file)
    if exists_file(file):
        return np.load(file, allow_pickle=allow_pickle, fix_imports=fix_imports)
    else:
        return None


def npyfile(filename):
    if filename.endswith('.npy'):
        return filename
    else:
        return '%s.npy' % filename


# Error
def last_error():
    return sys.exc_info()


def last_error_string():
    return format_error_ml(last_error())
    # return format_error(traceback.format_exc())


def format_error(err):
    try:
        parts = list(err[:2])
        parts.extend(traceback.format_tb(err[-1]))
        return parts
    except KeyboardInterrupt:
        raise
    except:
        return str(err)


def format_error_ml(err):
    return '\n'.join([str(v) for v in format_error(err)])


def print_stacktrace():
    traceback.print_exc()


def exit(code=0):
    os._exit(code)


# Data
def file_to_bytes(path, chunk_size=8192):
    return [b for b in bytes_stream(path, chunk_size)]


def bytes_stream(path, chunk_size=8192):
    with open(path, mode='rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if chunk:
                yield from chunk
            else:
                break


def bytes_to_string(v):
    return ''.join(map(chr, v))


def string_to_bytes(s):
    return [ord(c) for c in s]


def mask_for_number(num, min_size=3):
    return '%0{}d'.format(max(min_size, len(str(num))))


def dtype2str(dtype):
    if dtype == np.bool:
        return 'bool'
    elif dtype == np.int:
        return 'int'
    elif dtype == np.int8:
        return 'int8'
    elif dtype == np.int16:
        return 'int16'
    elif dtype == np.int32:
        return 'int32'
    elif dtype == np.int64:
        return 'int64'
    elif dtype == np.uint:
        return 'uint'
    elif dtype == np.uint8:
        return 'uint8'
    elif dtype == np.uint16:
        return 'uint16'
    elif dtype == np.uint32:
        return 'uint32'
    elif dtype == np.uint64:
        return 'uint64'
    elif dtype == np.float:
        return 'float'
    elif dtype == np.float16:
        return 'float16'
    elif dtype == np.float32:
        return 'float32'
    elif dtype == np.float64:
        return 'float64'
    else:
        raise Exception('Unknown type: ', dtype)


def str2dtype(dtype):
    if dtype == 'bool':
        return np.bool
    elif dtype == 'int':
        return np.int
    elif dtype == 'int8':
        return np.int8
    elif dtype == 'int16':
        return np.int16
    elif dtype == 'int32':
        return np.int32
    elif dtype == 'int64':
        return np.int64
    elif dtype == 'uint':
        return np.uint
    elif dtype == 'uint8':
        return np.uint8
    elif dtype == 'uint16':
        return np.uint16
    elif dtype == 'uint32':
        return np.uint32
    elif dtype == 'uint64':
        return np.uint64
    elif dtype == 'float':
        return np.float
    elif dtype == 'float16':
        return np.float16
    elif dtype == 'float32':
        return np.float32
    elif dtype == 'float64':
        return np.float64
    else:
        raise Exception('Unknown type: ', dtype)


class NamedEnum(enum.Enum):

    def __str__(self):
        return self._name_.lower()


# Numbers
def number_slot_max(num, slot, up_tick=True):
    return (num // slot) + 1 if up_tick else (num + slot - 1) // slot


def round_up_slot_number(num, slot, up_tick=True):
    return number_slot_max(num, slot, up_tick) * slot


def num2hformat(num):
    num = int(num)
    scale = np.log10(num)
    if scale >= 12:
        return '%dT' % (num // 1000000000000)
    elif scale >= 9:
        return '%dG' % (num // 1000000000)
    elif scale >= 6:
        return '%dM' % (num // 1000000)
    elif scale >= 3:
        return '%dk' % (num // 1000)
    else:
        return str(num)


def randomize(x):
    np.random.shuffle(x)
    return x


def seed(seed):
    return lambda id: np.random.seed(seed)


def escape_tf_run_name(run_name):
    return run_name.replace('[', '{').replace(']', '}')


# Multiprocess
def is_main_process():
    return mp.current_process().name == 'MainProcess'


def is_worker_process():
    return not is_main_process()


def getpid():
    return os.getpid()


def gettid():
    return threading.get_ident()


def gettname():
    return threading.current_thread().name


class SimpleFlag(object):

    def __init__(self, initval=False):
        self.value = initval

    def up(self):
        self.value = True
        return self

    def down(self):
        self.value = False
        return self

    def set(self, val=True):
        self.value = val
        return self


class SimpleCounter(object):

    def __init__(self, initval=0):
        self.value = initval

    def __eq__(self, other):
        return self.value == other

    def __le__(self, other):
        return self.value <= other

    def __lt__(self, other):
        return self.value < other

    def __ne__(self, other):
        return self.value != other

    def __gt__(self, other):
        return self.value > other

    def __ge__(self, other):
        return self.value >= other

    def __add__(self, other):
        return self.value + other

    def __iadd__(self, other):
        self.value += other
        return self

    def __sub__(self, other):
        return self.value - other

    def __isub__(self, other):
        self.value -= other
        return self

    def __str__(self):
        return str(self.value)

    def set(self, value=1):
        self.value = value
        return self

    def check(self, other=1):
        assert self.value == other

    def reset(self):
        self.value = 0
        return self

    def map_incr(self, v):
        self.value += v
        return v

    def map_count(self, v):
        self.value += 1
        return v

    def map_count_len(self, vs):
        self.value += len(vs)
        return vs

    def incr(self, num=1):
        self.value += num
        return self

    def decr(self, num=1):
        self.value -= num
        return self


class SharedMpCounter(object):

    def __init__(self, initval=0, ctx=mp.get_context('spawn')):
        self.val = ctx.RawValue('i', initval)
        self.lock = ctx.Lock()

    def incr(self, num=1):
        with self.lock:
            self.val.value += num
        return self

    def decr(self, num=1):
        with self.lock:
            self.val.value -= num
        return self

    def set(self, value=1):
        with self.lock:
            self.val.value = value
        return self

    @property
    def value(self):
        return self.val.value


class SharedMpBool(object):

    def __init__(self, initval=False, ctx=mp.get_context('spawn')):
        self.val = ctx.RawValue('b', initval)
        self.lock = ctx.Lock()

    def set(self, val=True):
        with self.lock:
            self.val.value = val
        return self

    def is_set(self):
        return self.value

    @property
    def value(self):
        return self.val.value


class SharedMtCounter(object):

    def __init__(self, initval=0):
        self.value = initval
        self.lock = threading.Lock()

    def incr(self, num=1):
        with self.lock:
            self.value += num
        return self

    def decr(self, num=1):
        with self.lock:
            self.value -= num
        return self

    def set(self, value=1):
        with self.lock:
            self.value = value
        return self


class SharedMtBool(object):

    def __init__(self, initval=False):
        self.value = initval
        self.lock = threading.Lock()

    def set(self, value=True):
        with self.lock:
            self.value = value
        return self

    def is_set(self):
        return self.value


def sleep(delay):
    time.sleep(delay)
