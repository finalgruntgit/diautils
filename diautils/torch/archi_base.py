from diautils import help
import torch
import torch.nn as nn
import numpy as np


def siglog(v):
    return v.sign() * torch.log(1 + v.abs())


def sigsqrt(v):
    return v / torch.sqrt(1 + v.abs())


def sigexp(v):
    sv = v.sign()
    return sv * (torch.exp(sv * v) - 1)


class FlattenModule(nn.Module):

    def forward(self, input):
        return input.reshape((input.shape[0], -1))


class ReshapeModule(nn.Module):

    def __init__(self, shape_out):
        super().__init__()
        self.shape_out = shape_out

    def forward(self, input):
        return input.view(self.shape_out)


class BackPropControlFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return siglog(grad_output)


class BackPropControlModule(nn.Module):

    def forward(self, v):
        return BackPropControlFunction.apply(v)


class SortModule(nn.Module):

    def __init__(self, shape_in, shape_sort=None, axis=-1):
        super().__init__()
        self.shape_in = shape_in
        self.shape_sort = shape_sort
        self.axis = axis

    def forward(self, v):
        if self.shape_sort is None:
            return v.sort(self.axis)
        else:
            return v.view(self.shape_sort).sort(self.axis)[0].view(self.shape_in)


class ZNormalizer(nn.Module):

    def __init__(self, input_shape=None, axis=None, beta=1e-6, epsilon=1e-12):
        super().__init__()
        self.input_shape = input_shape
        self.axis = axis
        if input_shape is not None and (isinstance(input_shape, tuple) or isinstance(input_shape, list)) and len(
                input_shape) > 1 and axis is not None and (
                (not isinstance(axis, tuple) and not isinstance(axis, list)) or len(axis) < len(input_shape)):
            if isinstance(axis, tuple) or isinstance(axis, list):
                self.output_shape = tuple(1 if i in axis else v for i, v in enumerate(input_shape))
            else:
                self.output_shape = tuple(1 if i == axis else v for i, v in enumerate(input_shape))
            self.output_shape = tuple(-1 if v is None else v for v in self.output_shape)
            self.param_shape = tuple(1 if (v is None or v == -1) else v for v in self.output_shape)
        else:
            self.output_shape = 1
            self.param_shape = 1
        self.beta = beta
        self.epsilon = epsilon
        if self.param_shape == 1:
            self.register_buffer('mean', torch.tensor(0.0))
            self.register_buffer('std', torch.tensor(1.0))
        else:
            self.register_buffer('mean', torch.zeros(self.param_shape))
            self.register_buffer('std', torch.ones(self.param_shape))
        self.register_buffer('beta_power', torch.tensor(1.0).double())
        self.disc = None

    def forward(self, v, idx=None):
        if self.training:
            with torch.no_grad():
                self.beta_power *= (1.0 - self.beta)
                ratio = (self.beta / torch.clamp(1.0 - self.beta_power, min=self.beta)).float()
                if idx is None:
                    v_ref = v
                else:
                    v_ref = v[idx]
                if self.output_shape == 1:
                    self.mean = (1.0 - ratio) * self.mean + ratio * v_ref.mean()
                    self.std = (1.0 - ratio) * self.std + ratio * ((v_ref - self.mean) ** 2).mean()
                else:
                    self.mean = (1.0 - ratio) * self.mean + ratio * v_ref.mean(self.axis).view(self.output_shape)
                    self.std = (1.0 - ratio) * self.std + ratio * ((v_ref - self.mean) ** 2).mean(self.axis).view(
                        self.output_shape)
                self.disc = 1.0 / (self.epsilon + torch.sqrt(self.std))
            if v.requires_grad:
                v = BackPropControlFunction.apply(v)
        elif self.disc is None:
            self.disc = 1.0 / (self.epsilon + torch.sqrt(self.std))
        return (v - self.mean) * self.disc


class SiglogZNormalizer(ZNormalizer):

    def __init__(self, input_shape=None, axis=None, beta=1e-6, epsilon=1e-12):
        super().__init__(input_shape, axis, beta, epsilon)

    def forward(self, v, idx=None):
        v_znorm = super().forward(v, idx)
        return v_znorm.sign() * torch.log(1.0 + v_znorm.abs())


class TanhSiglogZNormalizer(SiglogZNormalizer):

    def __init__(self, input_shape=None, axis=None, beta=1e-6, epsilon=1e-12, alpha=1.0):
        super().__init__(input_shape, axis, beta, epsilon)
        self.alpha = alpha

    def forward(self, v, idx=None):
        return (self.alpha * super().forward(v, idx)).tanh()


class BinaryNormalizer(TanhSiglogZNormalizer):

    def __init__(self, input_shape=None, axis=None, beta=1e-6, epsilon=1e-12, alpha=1.0):
        super().__init__(input_shape, axis, beta, epsilon, alpha)

    def forward(self, v, idx=None):
        return 0.5 * (1.0 + super().forward(v, idx))


class AxisSwapModule(nn.Module):

    def __init__(self, *axis):
        super().__init__()
        self.axis = axis

    def forward(self, v):
        return v.permute(*self.axis)


class SumModule(nn.Module):

    def __init__(self, *axis, keepdim=False):
        super().__init__()
        self.axis = axis
        self.keepdim = keepdim

    def forward(self, v):
        sum = v.sum(self.axis)
        if self.keepdim:
            dims = list(v.shape)
            if isinstance(self.axis, list) or isinstance(self.axis, tuple):
                for ax in self.axis:
                    dims[ax] = 1
            else:
                dims[self.axis] = 1
            sum = sum.view(dims)
        return sum


class MeanModule(nn.Module):

    def __init__(self, *axis, keepdim=False):
        super().__init__()
        self.axis = axis
        self.keepdim = keepdim

    def forward(self, v):
        mean = v.mean(self.axis)
        if self.keepdim:
            dims = list(v.shape)
            if isinstance(self.axis, list) or isinstance(self.axis, tuple):
                for ax in self.axis:
                    dims[ax] = 1
            else:
                dims[self.axis] = 1
            mean = mean.view(dims)
        return mean


class SiglogModule(nn.Module):

    def forward(self, v):
        return siglog(v)


class SigsqrtModule(nn.Module):

    def forward(self, v):
        return sigsqrt(v)


class SigmoidModule(nn.Module):

    def forward(self, v):
        return v.sigmoid()


class SoftmaxModule(nn.Module):

    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, v):
        return v.softmax(self.axis)


class L2NormalizerModule(nn.Module):

    def __init__(self, coef=1.0):
        super().__init__()
        self.coef = coef
        self.v = None

    def normalize(self, coef=1.0, retain_graph=True):
        if self.v is not None:
            loss = (self.v ** 2).mean() * (self.coef * coef)
            loss.backward(retain_graph=retain_graph)
            return loss
        else:
            return None

    def forward(self, v):
        self.v = v
        return v


class BaseArchi(nn.Module):

    def __init__(self):
        super().__init__()
        self.l2_norms = []

    def add_l2_norm(self, coef=1.0):
        l2norm = L2NormalizerModule(coef)
        self.l2_norms.append(l2norm)
        return l2norm

    def backward_l2_norms(self, coef=1.0, retain_graph=True):
        losses = []
        for norm in self.l2_norms:
            losses.append(norm.normalize(coef, retain_graph))
        return losses

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
        return FlattenModule()

    def reshape(self, shape_out):
        return ReshapeModule(shape_out)

    def znorm(self, input_shape=None, axis=None, beta=1e-6, epsilon=1e-12):
        return ZNormalizer(input_shape, axis, beta, epsilon)

    def siglog_znorm(self, input_shape=None, axis=None, beta=1e-6, epsilon=1e-12):
        return SiglogZNormalizer(input_shape, axis, beta, epsilon)

    def tanh_siglog_znorm(self, input_shape=None, axis=None, beta=1e-6, epsilon=1e-12, alpha=1.0):
        return TanhSiglogZNormalizer(input_shape, axis, beta, epsilon, alpha)

    def binary_norm(self, input_shape=None, axis=None, beta=1e-6, epsilon=1e-12, alpha=1.0):
        return BinaryNormalizer(input_shape, axis, beta, epsilon, alpha)

    def dense(self, input_size, output_size, bias=True):
        layer = nn.Linear(input_size, output_size, bias)
        nn.init.xavier_uniform_(layer.weight)
        if bias:
            nn.init.zeros_(layer.bias)
        return layer

    def conv1d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        nn.init.xavier_uniform_(layer.weight)
        if bias:
            nn.init.zeros_(layer.bias)
        return layer

    def conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        nn.init.xavier_uniform_(layer.weight)
        if bias:
            nn.init.zeros_(layer.bias)
        return layer

    def deconv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', output_padding=0):
        layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode)
        nn.init.xavier_uniform_(layer.weight)
        if bias:
            nn.init.zeros_(layer.bias)
        return layer

    def embedding(self, num_class, embedding_size):
        embedding = nn.Embedding(num_class, embedding_size)
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
        return nn.LeakyReLU(negative_slope, in_place)

    def tanh(self):
        return nn.Tanh()

    def backprop_control(self):
        return BackPropControlModule()

    def swap(self, *axis):
        return AxisSwapModule(*axis)

    def sum(self, *axis, keepdim=False):
        return SumModule(*axis, keepdim=keepdim)

    def mean(self, *axis, keepdim=False):
        return MeanModule(*axis, keepdim=keepdim)

    def siglog(self):
        return SiglogModule()

    def sigsqrt(self):
        return SigsqrtModule()

    def sigmoid(self):
        return SigmoidModule()

    def softmax(self, axis):
        return SoftmaxModule(axis)

    def sort(self, shape_in, shape_sort=None, axis=-1):
        return SortModule(shape_in, shape_sort, axis)

    def set_active(self, prefix, active=True):
        for name, param in self.named_parameters():
            if name.startswith(prefix):
                param.requires_grad = active
        return self
