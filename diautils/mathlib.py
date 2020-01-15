import numpy as np
from diautils import help
from diautils.config import to_conf
from bisect import bisect_left
from scipy.stats import norm


def siglog(v):
    return np.sign(v) * np.log(1.0 + np.abs(v))


def sigexp(v):
    sv = np.sign(v)
    return sv * (np.exp(sv * v) - 1.0)


def z_norm(v, mean=None, std=None):
    if len(v.shape) == 1:
        if mean is None:
            mean = v.mean()
        if std is None:
            std = v.std()
        np.where(std == 0, 1, std)
        return (v - mean) / std
    else:
        if mean is None:
            mean = v.mean(axis=-2)
        if std is None:
            std = v.std(axis=-2)
        np.where(std == 0, 1, std)
        return (v - np.expand_dims(mean, -2)) / np.expand_dims(std, -2)


def z_denorm(v, mean=None, std=None):
    if len(v.shape) == 1:
        return v * std + mean
    else:
        return (v.T * std + mean).T


def siglog_norm(v, mean=None, std=None):
    return siglog(z_norm(v, mean, std))


def siglog_denorm(v, mean=None, std=None):
    return z_denorm(sigexp(v), mean, std)


def tanh_siglog_norm(v, mean=None, std=None, alpha=1.0):
    return np.tanh(alpha * siglog_norm(v, mean, std))


def tanh_siglog_denorm(v, mean=None, std=None, alpha=1.0):
    return siglog_denorm(np.arctanh(v) / alpha, mean, std)


class Distrib1d:

    def __init__(self, sample, pre_norm, normal=None, mean=None, std=None, alpha=1.0):
        self.sample = sample
        self.lower_bound = sample[0]
        self.upper_bound = sample[-1]
        self.max_bound = np.maximum(np.abs(self.upper_bound), np.abs(self.lower_bound))
        self.bounds = (self.lower_bound, self.upper_bound)
        self.num_sample = len(sample)
        self.prob_step = 1 / (self.num_sample + 1)
        self.sample_probs = np.arange(1, self.num_sample + 1) * self.prob_step
        self.pre_norm = pre_norm
        if normal is None:
            self.normal = norm.ppf(self.sample_probs)
        else:
            self.normal = normal
        if self.pre_norm:
            self.lower_bound_normal = self.normal[1]
            self.upper_bound_normal = self.normal[-2]
        else:
            self.lower_bound_normal = self.normal[0]
            self.upper_bound_normal = self.normal[-1]
        self.max_bound_normal = np.maximum(np.abs(self.upper_bound_normal), np.abs(self.lower_bound_normal))
        self.bounds_normal = (self.lower_bound_normal, self.upper_bound_normal)
        self.mean = mean
        self.std = std
        self.alpha = alpha

    def clip(self, data):
        return np.clip(data, self.lower_bound, self.upper_bound)

    def clip_normal(self, data):
        return np.clip(data, self.lower_bound_normal, self.upper_bound_normal)

    def normalize(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if self.pre_norm:
            data = tanh_siglog_norm(data, self.mean, self.std, self.alpha)
        return data

    def denormalize(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if self.pre_norm:
            data = tanh_siglog_denorm(np.clip(data, -1 + 1e-15, 1 - 1e-15), self.mean, self.std, self.alpha)
        return data

    def search(self, data):
        data = self.normalize(data)
        pos = np.empty(len(data), np.int32)
        for i, v in enumerate(data):
            pos[i] = bisect_left(self.sample, v)
        return pos, data

    def probs(self, data):
        pos, data = self.search(data)
        spos = pos - 1
        lbound = self.sample[spos]
        ubound = self.sample[pos]
        dbound = ubound - lbound
        np.where(dbound == 0, 1, dbound)
        alphas = (data - lbound) / dbound
        return self.sample_probs[spos] + alphas * self.prob_step, data

    def as_normal(self, data, clip=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if clip:
            data = self.clip(data)
        probs, data_n = self.probs(data.flatten())
        return norm.ppf(probs).reshape(data.shape), data_n.reshape(data.shape)

    def as_raw(self, data, clip=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        data_flat = data.flatten()
        if clip:
            data_flat = self.clip_normal(data_flat)
        probs = norm.cdf(data_flat)
        pos_r = probs / self.prob_step
        pos = np.floor(pos_r).astype(int)
        alphas = pos_r - pos
        spos = pos - 1
        np.where(pos == len(self.sample), len(self.sample) - 1, pos)
        np.where(spos == -1, 0, spos)
        lbound = self.sample[spos]
        ubound = self.sample[pos]
        dbound = ubound - lbound
        data_normed = lbound + alphas * dbound
        data_normed = data_normed.reshape(data.shape)
        data_raw = self.denormalize(data_normed)
        return data_raw, data_normed

    def meta(self):
        return {
            'type': '1d',
            'pre_norm': self.pre_norm,
            'mean': self.mean,
            'std': self.std,
            'num_sample': self.num_sample,
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'prob_step': self.prob_step,
            'alpha': self.alpha
        }


class DistribNd:

    def __init__(self, sample, pre_norm, normal=None, mean=None, std=None, alpha=1.0):
        self.sample = sample
        self.lower_bound = sample.T[0]
        self.upper_bound = sample.T[-1]
        self.max_bound = np.maximum(np.abs(self.upper_bound), np.abs(self.lower_bound))
        self.bounds = np.stack((self.lower_bound, self.upper_bound), axis=-1)
        self.num_sample = sample.shape[-1]
        self.prob_step = 1 / (self.num_sample + 1)
        self.sample_probs = np.arange(1, self.num_sample + 1) * self.prob_step
        self.pre_norm = pre_norm
        if normal is None:
            self.normal = norm.ppf(self.sample_probs)
        else:
            self.normal = normal
        if self.pre_norm:
            self.lower_bound_normal = self.normal[1]
            self.upper_bound_normal = self.normal[-2]
        else:
            self.lower_bound_normal = self.normal[0]
            self.upper_bound_normal = self.normal[-1]
        self.max_bound_normal = np.maximum(np.abs(self.upper_bound_normal), np.abs(self.lower_bound_normal))
        self.bounds_normal = (self.lower_bound_normal, self.upper_bound_normal)
        self.mean = mean
        self.std = std
        self.alpha = alpha

    def clip(self, data):
        return np.clip(data.T, self.lower_bound, self.upper_bound).T

    def clip_normal(self, data):
        return np.clip(data.T, self.lower_bound_normal, self.upper_bound_normal).T

    def normalize(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if self.pre_norm:
            data = tanh_siglog_norm(data, self.mean, self.std, self.alpha)
        return data

    def denormalize(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if self.pre_norm:
            data = tanh_siglog_denorm(np.clip(data, -1 + 1e-15, 1 - 1e-15), self.mean, self.std, self.alpha)
        return data

    def search(self, data):
        data = self.normalize(data)
        data_flat = data.reshape((-1, data.shape[-1]))
        pos_flat = np.empty_like(data_flat, np.int32)
        sample_flat = self.sample.reshape((-1, self.sample.shape[-1]))
        for i, vs in enumerate(data_flat):
            for j, v in enumerate(vs):
                pos_flat[i, j] = bisect_left(sample_flat[i], v)
        return pos_flat.reshape(data.shape), data

    def probs(self, data):
        pos, data = self.search(data)
        spos = pos - 1
        sample_flat = self.sample.reshape((-1, self.sample.shape[-1]))
        num_iter = len(sample_flat)
        pos_flat = pos.reshape((num_iter, data.shape[-1]))
        spos_flat = spos.reshape((num_iter, data.shape[-1]))
        lbounds_flat = np.empty((num_iter, data.shape[-1]))
        ubounds_flat = np.empty((num_iter, data.shape[-1]))
        for i in range(num_iter):
            lbounds_flat[i] = sample_flat[i, spos_flat[i]]
            ubounds_flat[i] = sample_flat[i, pos_flat[i]]
        lbounds = lbounds_flat.reshape(data.shape)
        ubounds = ubounds_flat.reshape(data.shape)
        dbounds = ubounds - lbounds
        np.where(dbounds == 0, 1, dbounds)
        alphas = (data - lbounds) / dbounds
        return self.sample_probs[spos] + alphas * self.prob_step, data

    def as_normal(self, data, clip=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if clip:
            data = self.clip(data)
        probs, data_n = self.probs(data)
        return norm.ppf(probs), data_n

    def as_raw(self, data, clip=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if clip:
            data = self.clip_normal(data)
        probs = norm.cdf(data)
        pos_r = probs / self.prob_step
        pos = np.floor(pos_r).astype(int)
        alphas = pos_r - pos
        spos = pos - 1
        np.where(pos == len(self.sample), len(self.sample) - 1, pos)
        np.where(spos == -1, 0, spos)
        sample_flat = self.sample.reshape((-1, self.sample.shape[-1]))
        num_iter = len(sample_flat)
        pos_flat = pos.reshape((num_iter, data.shape[-1]))
        spos_flat = spos.reshape((num_iter, data.shape[-1]))
        lbounds_flat = np.empty((num_iter, data.shape[-1]))
        ubounds_flat = np.empty((num_iter, data.shape[-1]))
        for i in range(num_iter):
            lbounds_flat[i] = sample_flat[i, spos_flat[i]]
            ubounds_flat[i] = sample_flat[i, pos_flat[i]]
        lbounds = lbounds_flat.reshape(data.shape)
        ubounds = ubounds_flat.reshape(data.shape)
        dbounds = ubounds - lbounds
        data_normed = lbounds + alphas * dbounds
        data_raw = self.denormalize(data_normed)
        return data_raw, data_normed

    def meta(self):
        return {
            'type': 'nd',
            'pre_norm': self.pre_norm,
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
            'num_sample': self.num_sample,
            'lower_bound': self.lower_bound.tolist(),
            'upper_bound': self.upper_bound.tolist(),
            'prob_step': self.prob_step,
            'alpha': self.alpha
        }


class DistribMgr:

    def create_1d(self, data, num_sample=-1, bounds=None, alpha=1.0):
        if 0 < num_sample < len(data):
            sample = np.random.choice(data, num_sample)
        else:
            sample = data
            num_sample = len(data)
        sample = np.sort(sample)
        mu, sigma = sample.mean(), sample.std()
        if bounds is None:
            pre_norm = True
            sample = tanh_siglog_norm(sample, mu, sigma, alpha)
            bounds = (-1.0, 1.0)
        else:
            pre_norm = False
        bounded_sample = np.empty(num_sample + 2)
        bounded_sample[0] = bounds[0]
        bounded_sample[-1] = bounds[1]
        bounded_sample[1:-1] = sample
        return Distrib1d(bounded_sample, pre_norm, None, mu, sigma, alpha)

    def create_nd(self, data, num_sample=-1, bounds=None, alpha=1.0):
        if 0 < num_sample < data.shape[-1]:
            data_flat = data.reshape((-1, data.shape[-1]))
            sample = np.empty((len(data_flat), num_sample), data.dtype)
            for i, vs in enumerate(data_flat):
                sample[i] = np.random.choice(vs, num_sample)
            sample = sample.reshape(data.shape[:-1] + (num_sample,))
        else:
            sample = data
            num_sample = data.shape[-1]
        sample = np.sort(sample)
        mu, sigma = sample.mean(axis=-1), sample.std(axis=-1)
        if bounds is None:
            pre_norm = True
            sample = tanh_siglog_norm(sample, mu, sigma, alpha)
            bounds = np.tile((-1.0, 1.0), data.shape[:-1] + (1,))
        else:
            pre_norm = False
        bounded_sample = np.empty((num_sample + 2,) + data.shape[:-1])
        bounded_sample[0] = bounds[:, :, 0]
        bounded_sample[-1] = bounds[:, :, 1]
        bounded_sample[1:-1] = sample.T
        bounded_sample = bounded_sample.T
        return DistribNd(bounded_sample, pre_norm, None, mu, sigma, alpha)

    def create(self, data, num_sample=-1, bounds=None, alpha=1.0, flatten=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if len(data.shape) == 1:
            return self.create_1d(data, num_sample, bounds, alpha)
        elif flatten:
            return self.create_1d(data.flatten(), num_sample, bounds, alpha)
        else:
            return self.create_nd(data, num_sample, bounds, alpha)

    def as_normal_1d(self, data):
        num_data = len(data)
        lin = np.arange(1, num_data + 1) / (num_data + 1)
        data_normal = np.empty_like(data, np.float)
        lin_normal = norm.ppf(lin)
        ss = np.argsort(data)
        data_normal[ss] = lin_normal
        return data_normal

    def as_normal_nd(self, data):
        data_flat = data.reshape((-1, data.shape[-1]))
        data_normal = np.empty_like(data_flat, np.float)
        for i, vs in enumerate(data_flat):
            data_normal[i] = self.as_normal_1d(vs)
        return data_normal.reshape(data.shape)

    def as_normal(self, data, flatten=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if len(data.shape) == 1:
            return self.as_normal_1d(data)
        elif flatten:
            return self.as_normal_1d(data.flatten()).reshape(data.shape)
        else:
            return self.as_normal_nd(data)

    def save(self, distrib, name, dir_data='data'):
        help.save_npy(help.join(dir_data, '{}_sample'.format(name)), distrib.sample)
        help.save_npy(help.join(dir_data, '{}_normal'.format(name)), distrib.normal)
        help.save_json_pretty(help.join(dir_data, '{}_meta.json'.format(name)), distrib.meta())
        return distrib

    def load(self, name, dir_data='data'):
        sample = help.load_npy(help.join(dir_data, '{}_sample'.format(name)))
        normal = help.load_npy(help.join(dir_data, '{}_normal'.format(name)))
        meta = to_conf().load(help.join(dir_data, '{}_meta.json'.format(name)))
        dist_type = meta['type']
        if dist_type == '1d':
            return Distrib1d(sample, meta['pre_norm'], normal=normal, mean=meta['mean'], std=meta['std'], alpha=meta['alpha', 1.0])
        elif dist_type == 'nd':
            return DistribNd(sample, meta['pre_norm'], normal=normal, mean=meta['mean'], std=meta['std'], alpha=meta['alpha', 1.0])
        else:
            raise Exception('Unknown distribution type: {}'.format(dist_type))


def bjorck1(m, steps):
    ident = np.eye(len(m.T))
    for i in range(steps):
        m = np.matmul(m, ident + 0.5 * (ident - np.matmul(m.T, m)))
    return m
