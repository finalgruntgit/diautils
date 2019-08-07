import numpy as np
from diautils.stream import pipe
from diautils import help


class DatasetType(help.NamedEnum):
    TRAIN = 0,
    TEST = 1,
    RAW = 2


class Dataset:

    def __init__(self, ds_name, ds_type, ds_field, data, dir_base, dir_name, dir_version, dir_type, file_data, file_meta):
        self.data = data
        self.name = ds_name
        self.type = ds_type
        self.field = ds_field
        self.shape = data.shape
        self.dtype = data.dtype
        self.dir_base = dir_base
        self.dir_name = dir_name
        self.dir_version = dir_version
        self.dir_type = dir_type
        self.file_data = file_data
        self.file_meta = file_meta

    def fill(self, value):
        self.data.fill(value)
        return self

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

    def __lt__(self, other):
        return self.data.__lt__(other)

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

    def symlink(self, dir_destination, add_cwd=True, meta_suffix='.meta'):
        ds_sym_clone(self.dir_type, dir_destination, self.field, add_cwd, meta_suffix)
        return self


class DatasetBuilder:

    def __init__(self, ds_name, ds_version, ds_type, ds_field, dir_base='data'):
        self.name = ds_name
        self.version = ds_version
        self.type = ds_type
        self.field = ds_field
        self.dir_base = dir_base

    def exists(self, meta_suffix='.meta'):
        dirs = self.dirs()
        for d in dirs:
            if not help.exists_file(d):
                return False
        file_data = help.join(dirs[-1], str(self.field))
        file_meta = '{}{}'.format(file_data, meta_suffix)
        return help.exists_file(file_meta)

    def symlink(self, dir_destination, add_cwd=True, meta_suffix='.meta'):
        dir_ds = help.join(self.dir_base, str(self.name), str(self.type))
        ds_sym_clone(dir_ds, dir_destination, self.field, add_cwd, meta_suffix)
        return self

    def dirs(self, index=None):
        values = ds_dirs(self.dir_base, self.name, self.version, self.type)
        if index is None:
            return values
        else:
            return values[index]

    def create(self, shape, dtype=np.float64, mode='w+', meta_suffix='.meta'):
        if isinstance(shape, int):
            shape = (shape,)
        else:
            shape = tuple([int(v) for v in shape])
        _, dir_name, dir_version, dir_type = self.dirs()
        help.mkdirs(dir_type)
        file_data = help.join(dir_type, str(self.field))
        data = np.memmap(file_data, dtype, shape=tuple(shape), mode=mode)
        meta = {
            'type': help.dtype2str(dtype),
            'shape': shape
        }
        file_meta = '{}{}'.format(file_data, meta_suffix)
        help.save_json_pretty(file_meta, meta)
        return Dataset(self.name, self.type, self.field, data, self.dir_base, dir_name, dir_version, dir_type, file_data, file_meta)

    def create_from(self, data, dtype=None, mode='w+', meta_suffix='.meta'):
        dataset = self.create(data.shape, data.dtype if dtype is None else dtype, mode, meta_suffix)
        dataset[:] = data
        return dataset

    def load(self, mode='r', meta_suffix='.meta'):
        _, dir_name, dir_version, dir_type = self.dirs()
        file_data = help.join(dir_type, str(self.field))
        file_meta = '{}{}'.format(file_data, meta_suffix)
        meta = help.load_json(file_meta)
        shape = tuple(meta['shape'])
        dtype = help.str2dtype(meta['type'])
        data = np.memmap(file_data, dtype, shape=shape, mode=mode)
        return Dataset(self.name, self.type, self.field, data, self.dir_base, dir_name, dir_version, dir_type, file_data, file_meta)


class DatasetManager:

    def __init__(self, ds_name, ds_version=None, ds_type=None, dir_base='data'):
        self.name = ds_name
        self.version = ds_version
        self.type = ds_type
        self.dir_base = dir_base

    def dirs(self, index=None):
        values = ds_dirs(self.dir_base, self.name, self.version, self.type)
        if index is None:
            return values
        else:
            return values[index]

    def init_ds(self, dsfield):
        return ds(self.name, self.version, self.type, dsfield, self.dir_base)

    def create_ds(self, dsfield, shape, dtype=np.float64, mode='w+', meta_suffix='.meta'):
        return self.init_ds(dsfield).create(shape, dtype, mode, meta_suffix)

    def create_ds_from(self, dsfield, data, dtype=None, mode='w+', meta_suffix='.meta'):
        return self.init_ds(dsfield).create_from(data, mode, meta_suffix)

    def edit_ds(self, dsfield, meta_suffix='.meta'):
        return self.load_ds(dsfield, 'r+', meta_suffix)

    def load_ds(self, dsfield, mode='r', meta_suffix='.meta'):
        return self.init_ds(dsfield).load(mode, meta_suffix)


def ds_sym_clone(dir_src, dir_destination, dsfield, add_cwd=True, meta_suffix='.meta'):
    file_data = help.join(dir_src, str(dsfield))
    file_data_out = help.join(dir_destination, str(dsfield))
    help.symlink(file_data, file_data_out, add_cwd)
    file_meta = '{}{}'.format(file_data, meta_suffix)
    file_meta_out = '{}{}'.format(file_data_out, meta_suffix)
    help.symlink(file_meta, file_meta_out, add_cwd)


def ds_dirs(dir_base, ds_name, ds_version, ds_type):
    if ds_name:
        dir_name = help.join(dir_base, str(ds_name))
    else:
        dir_name = dir_base
    if ds_version:
        dir_version = help.join(dir_name, str(ds_version))
    else:
        dir_version = dir_name
    if ds_type:
        dir_type = help.join(dir_version, str(ds_type))
    else:
        dir_type = dir_version
    return dir_base, dir_name, dir_version, dir_type


def ds(ds_name, ds_version, ds_type, ds_field, dir_base='data'):
    return DatasetBuilder(ds_name, ds_version, ds_type, ds_field, dir_base)
