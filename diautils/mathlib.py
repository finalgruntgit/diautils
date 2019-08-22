import numpy as np
from diautils import help
from diautils.config import to_conf
from bisect import bisect_left
from scipy.stats import norm


def siglog(v):
    return np.sign(v) * np.log(1 + np.abs(v))


def sigexp(v):
    sv = np.sign(v)
    return sv * (np.exp(sv * v) - 1)


def z_norm(v, mean=None, std=None):
    if mean is None:
        mean = v.mean(axis=0)
    if std is None:
        std = v.std(axis=0)
    np.where(std == 0, 1, std)
    return (v - mean) / std


def z_denorm(v, mean=None, std=None):
    return v * std + mean


def siglog_norm(v, mean=None, std=None):
    return siglog(z_norm(v, mean, std))


def siglog_denorm(v, mean=None, std=None):
    return z_denorm(sigexp(v), mean, std)


def tanh_siglog_norm(v, mean=None, std=None, alpha=1.0):
    return np.tanh(alpha * siglog_norm(v, mean, std))


def tanh_siglog_denorm(v, mean=None, std=None, alpha=1.0):
    return siglog_denorm(np.arctanh(v) / alpha, mean, std)


class Distrib:

    def __init__(self, sample, pre_norm, normal=None, mean=None, std=None, alpha=1.0):
        self.sample = sample
        self.lower_bound = sample[0]
        self.upper_bound = sample[-1]
        self.max_bound = max(self.upper_bound, abs(self.lower_bound))
        self.bounds = (self.lower_bound, self.upper_bound)
        self.num_sample = len(sample)
        self.prob_step = 1 / (self.num_sample + 1)
        self.sample_probs = np.arange(1, self.num_sample + 1) * self.prob_step
        if normal is None:
            self.normal = norm.ppf(self.sample_probs)
        else:
            self.normal = normal
        self.pre_norm = pre_norm
        self.mean = mean
        self.std = std
        self.alpha = alpha

    def normalize(self, data):
        if self.pre_norm is not None:
            data = tanh_siglog_norm(data, self.mean, self.std, self.alpha)
        return data

    def denormalize(self, data):
        if self.pre_norm is not None:
            data = tanh_siglog_denorm(data, self.mean, self.std, self.alpha)
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

    def interpolate(self, data):
        probs, data = self.probs(data)
        return norm.ppf(probs), data

    def interpolate_inv(self, data):
        probs = norm.cdf(data)
        pos_r = probs / self.prob_step
        pos = np.floor(pos_r).astype(int)
        alphas = pos_r - pos
        spos = pos - 1
        lbound = self.sample[spos]
        ubound = self.sample[pos]
        dbound = ubound - lbound
        data_normed = lbound + alphas * dbound
        data = self.denormalize(data_normed)
        return data, data_normed

    def meta(self):
        return {
            'pre_norm': self.pre_norm,
            'mean': self.mean,
            'std': self.std,
            'num_sample': self.num_sample,
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'prob_step': self.prob_step,
            'alpha': self.alpha
        }


class DistribMgr:

    def create(self, data, num_sample=-1, bounds=None, alpha=1.0):
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
        return Distrib(bounded_sample, pre_norm, mean=mu, std=sigma, alpha=alpha)

    def as_normal_1d(self, data):
        num_data = len(data)
        lin = np.arange(1, num_data + 1) / (num_data + 1)
        data_normal = np.empty_like(data, np.float)
        lin_normal = norm.ppf(lin)
        ss = np.argsort(data)
        data_normal[ss] = lin_normal
        return data_normal

    def as_normal(self, data, flatten=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if len(data.shape) == 1:
            return self.as_normal_1d(data)
        else:
            data_flat = np.reshape(data, (-1, data.shape[-1]))
            if flatten:
                data_normal = self.as_normal_1d(data_flat)
            else:
                data_normal = np.empty_like(data_flat, np.float)
                for i, vs in enumerate(data_flat):
                    data_normal[i] = self.as_normal_1d(vs)
            return np.reshape(data_normal, data.shape)

    def save(self, distrib, name, dir_data='data'):
        help.save_npy(help.join(dir_data, '{}_sample'.format(name)), distrib.sample)
        help.save_npy(help.join(dir_data, '{}_normal'.format(name)), distrib.normal)
        help.save_json_pretty(help.join(dir_data, '{}_meta.json'.format(name)), distrib.meta())
        return distrib

    def load(self, name, dir_data='data'):
        sample = help.load_npy(help.join(dir_data, '{}_sample'.format(name)))
        normal = help.load_npy(help.join(dir_data, '{}_normal'.format(name)))
        meta = to_conf().load(help.join(dir_data, '{}_meta.json'.format(name)))
        return Distrib(sample, meta['pre_norm'], normal=normal, mean=meta['mean'], std=meta['std'], alpha=meta['alpha', 1.0])