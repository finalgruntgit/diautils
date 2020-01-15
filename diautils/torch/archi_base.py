from diautils import help
import torch
import torch.nn as nn


class Flatten(nn.Module):

    def forward(self, input):
        return input.view((input.shape[0], -1))


class ZNormalizer(nn.Module):

    def __init__(self, input_shape=None, axis=None, beta=1e-6, epsilon=1e-12):
        super().__init__()
        self.input_shape = input_shape
        self.axis = axis
        if input_shape is not None and (isinstance(input_shape, tuple) or isinstance(input_shape, list)) and len(input_shape) > 1 and axis is not None and ((not isinstance(axis, tuple) and not isinstance(axis, list)) or len(axis) < len(input_shape)):
            if isinstance(axis, tuple) or isinstance(axis, list):
                self.output_shape = tuple(1 if i in axis else v for i, v in enumerate(input_shape))
            else:
                self.output_shape = tuple(1 if i == axis else v for i, v in enumerate(input_shape))
        else:
            self.output_shape = 1
        self.beta = beta
        self.epsilon = epsilon
        if self.output_shape == 1:
            self.register_buffer('mean', torch.tensor(0.0))
            self.register_buffer('std', torch.tensor(1.0))
        else:
            self.register_buffer('mean', torch.zeros(self.output_shape))
            self.register_buffer('std', torch.ones(self.output_shape))
        self.register_buffer('beta_power', torch.tensor(1.0))
        self.register_buffer('ratio', torch.tensor(1.0))
        self.disc = self.epsilon + torch.sqrt(self.std)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if len(args) == 1 and isinstance(args[0], torch.device):
            device = args[0]
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
            self.beta_power = self.beta_power.to(device)
            self.ratio = self.ratio.to(device)
            self.disc = self.disc.to(device)
        return self

    def forward(self, v, idx=None):
        if self.training:
            with torch.no_grad():
                self.beta_power *= (1.0 - self.beta)
                ratio = self.beta / (1.0 - self.beta_power)
                if idx is None:
                    v_ref = v
                else:
                    v_ref = v[idx]
                if self.output_shape == 1:
                    self.mean = (1.0 - ratio) * self.mean + ratio * v_ref.mean()
                    self.std = (1.0 - ratio) * self.std + ratio * ((v_ref - self.mean) ** 2).mean()
                else:
                    self.mean = (1.0 - ratio) * self.mean + ratio * v_ref.mean(self.axis).view(self.output_shape)
                    self.std = (1.0 - ratio) * self.std + ratio * ((v_ref - self.mean) ** 2).mean(self.axis).view(self.output_shape)
                self.disc = self.epsilon + torch.sqrt(self.std)
        return (v - self.mean) / self.disc


class SiglogZNormalizer(ZNormalizer):

    def __init__(self, input_shape=None, axis=None, beta=1e-6, epsilon=1e-12):
        super().__init__(input_shape, axis, beta, epsilon)

    def forward(self, v, idx=None):
        v_znorm = super().forward(v, idx)
        return v_znorm.sign() * torch.log(1.0 + v_znorm.abs())


class TanhSiglogZNormalizer(SiglogZNormalizer):

    def __init__(self, input_shape=None, axis=None, beta=1e-6, epsilon=1e-12):
        super().__init__(input_shape, axis, beta, epsilon)

    def forward(self, v, idx=None):
        return super().forward(v, idx).tanh()


class BaseArchi(nn.Module):

    def siglog(self, v):
        return v.sign() * torch.log(1 + v.abs())

    def sigexp(self, v):
        sv = v.sign()
        return sv * (torch.exp(sv * v) - 1)

    def activation(self, name):
        if name == 'none':
            return lambda v: v
        elif name == 'lrelu':
            return nn.LeakyReLU()
        elif name == 'selu':
            return nn.SELU()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'siglog':
            return self.siglog
        else:
            raise Exception('Unsupported activation function: {}'.format(name))

    def flatten(self):
        return Flatten()

    def znorm(self, input_shape=None, axis=None, beta=1e-6, epsilon=1e-12):
        return ZNormalizer(input_shape, axis, beta, epsilon).to(self.device)

    def siglog_znorm(self, input_shape=None, axis=None, beta=1e-6, epsilon=1e-12):
        return SiglogZNormalizer(input_shape, axis, beta, epsilon).to(self.device)

    def tanh_siglog_znorm(self, input_shape=None, axis=None, beta=1e-6, epsilon=1e-12):
        return TanhSiglogZNormalizer(input_shape, axis, beta, epsilon).to(self.device)

    def dense(self, input_size, output_size, bias=True):
        layer = nn.Linear(input_size, output_size, bias)
        nn.init.xavier_uniform_(layer.weight)
        if bias:
            nn.init.zeros_(layer.bias)
        return layer

    def embedding(self, num_class, embedding_size):
        embedding = nn.Embedding(num_class, embedding_size).to(self.device)
        nn.init.xavier_uniform_(embedding.weight)
        return embedding

    def display_params(self):
        num_param = 0
        for name, parameter in self.named_parameters():
            print('{} => {}'.format(name, help.human_readable(parameter.numel())))
            num_param += parameter.numel()
        print('Total number of parameter: {}'.format(help.human_readable(num_param)))
        return self

    def lrelu(self, negative_slope=0.2, in_place=False):
        return nn.LeakyReLU(negative_slope, in_place).to(self.device)
