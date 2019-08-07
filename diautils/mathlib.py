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
        mean = v.mean()
    if std is None:
        std = v.std()
    np.where(std == 0, 1, std)
    return (v - mean) / std


def siglog_norm(v, mean=None, std=None):
    return siglog(z_norm(v, mean, std))


def tanh_siglog_norm(v, mean=None, std=None):
    return np.tanh(siglog_norm(v, mean, std))


class Distrib:

    def __init__(self, sample, pre_norm, normal=None, mean=None, std=None):
        self.sample = sample
        self.lower_bound = sample[0]
        self.upper_bound = sample[-1]
        self.bounds = (self.lower_bound, self.upper_bound)
        self.num_sample = len(sample)
        self.prob_step = 1 / (self.num_sample + 3)
        self.sample_probs = np.arange(1, self.num_sample + 3) * self.prob_step
        if normal is None:
            self.normal = norm.ppf(self.sample_probs)
        else:
            self.normal = normal
        self.pre_norm = pre_norm
        self.mean = mean
        self.std = std

    def search(self, data):
        if self.pre_norm is not None:
            data = tanh_siglog_norm(data, self.mean, self.std)
        pos = np.empty(len(data), np.int32)
        for i, v in enumerate(data):
            pos[i] = bisect_left(self.sample, v)
        return pos, data

    def interpolate(self, data):
        pos, data = self.search(data)
        spos = pos - 1
        lbound = self.sample[spos]
        ubound = self.sample[pos]
        dbound = ubound - lbound
        np.where(dbound == 0, 1, dbound)
        alphas = (data - lbound) / dbound
        return spos, pos, alphas, data

    def probs(self, data):
        spos, pos, alphas, data = self.interpolate(data)
        return self.sample_probs[spos] + alphas * self.prob_step, data

    def interpolate_exact(self, data):
        probs, data = self.probs(data)
        return norm.ppf(probs), data

    def interpolate_linear(self, data):
        spos, pos, alphas, data = self.interpolate(data)
        lbound_normal = self.normal[spos]
        ubound_normal = self.normal[pos]
        dbound_normal = ubound_normal - lbound_normal
        return self.normal[spos] + alphas * dbound_normal, data

    def meta(self):
        return {
            'pre_norm': self.pre_norm,
            'mean': self.mean,
            'std': self.std,
            'num_sample': self.num_sample,
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'prob_step': self.prob_step,
        }


class DistribMgr:

    def create(self, data, num_sample=-1, bounds=None):
        if 0 < num_sample < len(data):
            sample = np.random.choice(data, num_sample)
        else:
            sample = data
            num_sample = len(data)
        sample = np.sort(sample)
        mu, sigma = sample.mean(), sample.std()
        if bounds is None:
            pre_norm = True
            sample = tanh_siglog_norm(sample, mu, sigma)
            bounds = (-1.0, 1.0)
        else:
            pre_norm = False
        bounded_sample = np.empty(num_sample + 2)
        bounded_sample[0] = bounds[0]
        bounded_sample[-1] = bounds[1]
        bounded_sample[1:-1] = sample
        return Distrib(bounded_sample, pre_norm, mean=mu, std=sigma)

    def normalize(self, data):
        num_data = len(data)
        lin = np.arange(1, num_data + 1) / (num_data + 1)
        lin_normal = norm.ppf(lin)
        ss = np.argsort(data)
        data_normal = np.empty(num_data)
        data_normal[ss] = lin_normal
        return data_normal

    def save(self, distrib, name, dir_data='data'):
        help.save_npy(help.join(dir_data, '{}_sample'.format(name)), distrib.sample)
        help.save_npy(help.join(dir_data, '{}_normal'.format(name)), distrib.normal)
        help.save_json_pretty(help.join(dir_data, '{}_meta.json'.format(name)), distrib.meta())
        return distrib

    def load(self, name, dir_data='data'):
        sample = help.load_npy(help.join(dir_data, '{}_sample'.format(name)))
        normal = help.load_npy(help.join(dir_data, '{}_normal'.format(name)))
        meta = to_conf().load(help.join(dir_data, '{}_meta.json'.format(name)))
        return Distrib(sample, meta['pre_norm'], normal=normal, mean=meta['mean'], std=meta['std'])
