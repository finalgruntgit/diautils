from diautils import help
from diautils.config import to_conf
from diautils.stream import pipe
import numpy as np


class Dataset:

    def __init__(self, name, dir='', meta_suffix='meta'):
        self.name = name
        self.dir = dir
        self.meta_suffix = meta_suffix
        self.file_data = help.join(self.dir, name)
        self.file_meta = help.join(self.dir, '{}.{}'.format(name, self.meta_suffix))
        self.meta = None
        self.data = None
        self.shape = None
        self.dtype = None

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)

    def __len__(self):
        return self.data.__len__()

    def __eq__(self, other):
        return self.data.__eq__(other)

    def __ne__(self, other):
        return self.data.__ne__(other)

    def __gt__(self, other):
        return self.data.__gt__(other)

    def __ge__(self, other):
        return self.data.__ge__(other)

    def __lt__(self, other):
        return self.data.__lt__(other)

    def __le__(self, other):
        return self.data.__le__(other)

    def __add__(self, other):
        return self.data.__add__(other)

    def __sub__(self, other):
        return self.data.__sub__(other)

    def __and__(self, other):
        return self.data.__and__(other)

    def __cmp__(self, other):
        return self.data.__cmp__(other)

    def __mul__(self, other):
        return self.data.__mul__(other)

    def fill(self, value):
        self.data.fill(value)
        return self

    def set(self, data):
        self.data[:] = data
        return self

    def pipe(self):
        return pipe(self.data)

    def shuffle(self, perm=None):
        if perm is None:
            np.random.shuffle(self.data)
        else:
            self.data[:] = self.data[perm]

    def transform(self, tfunc):
        self.data = tfunc(self.data)
        self.shape = self.data.shape
        self.dtype = self.data.dtype
        return self

    def load(self, mode='r'):
        if help.exists_file(self.file_data):
            self.meta = to_conf().load(self.file_meta)
            self.shape = tuple(self.meta['shape'])
            self.dtype = help.str2dtype(self.meta['type', 'float64'])
            self.data = np.memmap(self.file_data, self.dtype, shape=self.shape, mode=mode)
        return self

    def create(self, shape, dtype=np.float64, mode='w+'):
        if isinstance(shape, int):
            self.shape = (shape, )
        else:
            self.shape = tuple(int(v) for v in shape)
        self.dtype = dtype
        help.mkdirs(self.dir)
        self.data = np.memmap(self.file_data, self.dtype, shape=self.shape, mode=mode)
        self.meta = to_conf({
            'type': help.dtype2str(self.dtype),
            'shape': list(self.shape)
        })
        self.meta.save(self.file_meta)
        return self

    def create_like(self, data, dtype=None, mode='w+'):
        return self.create(data.shape, data.dtype if dtype is None else dtype, mode)

    def create_from(self, data, dtype=None, mode='w+'):
        return self.create(data.shape, data.dtype if dtype is None else dtype, mode).set(data)

    def flatten(self):
        return self.data.flatten()


def ds_init(name, dir='', meta_suffix='meta'):
    return Dataset(name, dir, meta_suffix)


def ds_load(name, dir='', meta_suffix='meta', mode='r'):
    return ds_init(name, dir, meta_suffix).load(mode)


def ds_create(name, shape, dtype=np.float64, dir='', meta_suffix='meta', mode='w+'):
    return ds_init(name, dir, meta_suffix).create(shape, dtype, mode)


def ds_create_like(name, data, dtype=None, dir='', meta_suffix='meta', mode='w+'):
    return ds_init(name, dir, meta_suffix).create_like(data, dtype, mode)


def ds_create_from(name, data, dtype=None, dir='', meta_suffix='meta', mode='w+'):
    return ds_init(name, dir, meta_suffix).create_from(data, dtype, mode)


class DatasetType(help.NamedEnum):
    TRAIN = 0,
    TEST = 1,
    RAW = 2,
    VALID = 3


class DatasetManager:

    def __init__(self, ds_name, ds_version=None, ds_type=None, dir_base='data'):
        self.name = ds_name
        self.version = ds_version
        self.type = ds_type
        self.dir_base = dir_base
        if self.name:
            self.dir_name = help.join(self.dir_base, str(self.name))
        else:
            self.dir_name = self.dir_base
        if self.version:
            self.dir_version = help.join(self.dir_name, str(self.version))
        else:
            self.dir_version = self.dir_name
        if self.type:
            self.dir_type = help.join(self.dir_version, str(self.type))
        else:
            self.dir_type = self.dir_version

    def init_ds(self, dsfield):
        return ds_init(dsfield, self.dir_type)
