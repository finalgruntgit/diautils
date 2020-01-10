from diautils import help


class Config():

    def __init__(self, ini=None):
        self.data = {}
        self.path_sep = '.'
        if ini is not None:
            self.update(ini)

    def path_separator(self, path_sep):
        self.path_sep = path_sep
        return self

    def clear(self):
        self.data.clear()
        return self

    def copy(self):
        return Config(self.data.copy())

    def items(self):
        return self.data.items()

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def get_parent(self, k):
        if isinstance(k, str):
            paths = k.split(self.path_sep)
        elif isinstance(k, list):
            paths = k
        else:
            return self.data, k

        if len(paths) == 1:
            return self.data, paths[0]
        else:
            node = self.data
            for path in paths[:-1]:
                node = node.get(path)
                if node is None or not isinstance(node, dict):
                    return None, k
            return node, paths[-1]

    def get_or_create_parent(self, k):
        if isinstance(k, str):
            paths = k.split(self.path_sep)
        elif isinstance(k, list):
            paths = k
        else:
            return self.data, k

        if len(paths) == 1:
            return self.data, paths[0]
        else:
            node = self.data
            for path in paths[:-1]:
                child = node.get(path)
                if child is None:
                    child = {}
                    node[path] = child
                else:
                    assert isinstance(child, dict)
                node = child
            return node, paths[-1]

    def pop(self, k, d=None):
        node, k = self.get_parent(k)
        return d if node is None else node.pop(k, d)

    def put_if_absent(self, k, d=None):
        node, k = self.get_or_create_parent(k)
        return node.setdefault(k, d)

    def get(self, k, d=None):
        v = self.__getitem__(k)
        return d if v is None else v

    def put(self, k, v):
        self.__setitem__(k, v)
        return self

    def __getitem__(self, k):
        if isinstance(k, tuple):
            return self.get(k[0], k[1])
        else:
            node, k = self.get_parent(k)
            if node is None:
                return None
            else:
                try:
                    return node.__getitem__(k)
                except:
                    return None

    def __setitem__(self, k, v):
        node, k = self.get_or_create_parent(k)
        node.__setitem__(k, v)

    def __contains__(self, k):
        node, k = self.get_parent(k)
        return False if node is None else node.__contains__(k)

    def __iter__(self):
        return self.data.__iter__()

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return self.pretty()

    def update(self, other):
        if other is None:
            return self
        elif isinstance(other, dict):
            help.merge_write(self.data, other)
        elif isinstance(other, tuple):
            self.put(other[0], other[1])
        elif isinstance(other, list):
            for row in other:
                self.put(row[0], row[1])
        return self

    def update_all(self, others):
        for other in others:
            self.update(other)
        return self

    def load(self, file, zip=False):
        return self.update(help.load_json(file, zip=zip))

    def save(self, file, indent=4, zip=False):
        help.save_json(file, self.data, indent=indent, zip=zip)
        return self

    def conf(self, k, d=None):
        if d is None:
            d = {}
        return Config(self.get(k, d))

    def pretty(self, indent=4):
        return help.format_json_pretty(self.data, indent)

    def steal(self, other_config, k, d=None):
        if d is None:
            d = {}
        return self.put(k, other_config.get(k, d))


def to_conf(ini=None):
    return ini if ini is not None and isinstance(ini, Config) else Config(ini)
