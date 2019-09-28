import numpy as np


def calc_conv_sizes(initial_size, convs):
    sizes = [initial_size]
    for cursor, conv in enumerate(convs):
        sizes.append(int(np.ceil(sizes[cursor] / conv[1])))
    return sizes


def calc_receptive_field(sizes, convs, pos):
    num_layers = len(convs)
    left = pos
    right = pos
    cursor = num_layers - 1
    for conv in convs[::-1]:
        k = (conv[0] - 1) // 2
        s = conv[1]
        left = max(0, left * s - k)
        right = min(sizes[cursor] - 1, right * s + k)
        cursor -= 1
    return left, right


def calc_receptive_field_matrix(initial_size, convs, return_sizes=False):
    sizes = calc_conv_sizes(initial_size, convs)
    final_size = sizes[-1]
    matrix = np.zeros((final_size, initial_size), np.float32)
    for i in range(final_size):
        l, r = calc_receptive_field(sizes, convs, i)
        matrix[i, l: r + 1] = 1.0
    counts = np.sum(matrix, axis=0)
    np.divide(matrix, counts, matrix)
    if return_sizes:
        return matrix, sizes
    else:
        return matrix


def calc_attention(initial_size, convs, coefs, return_sizes=False):
    if return_sizes:
        matrix, sizes = calc_receptive_field_matrix(initial_size, convs, True)
        return np.matmul(coefs, matrix), sizes
    else:
        return np.matmul(coefs, calc_receptive_field_matrix(initial_size, convs, False))


def calc_param_count(initial_shape, convs):
    if len(initial_shape) == 2:
        channels = initial_shape[1]
    else:
        channels = 1
    num = 0
    for conv in convs:
        num += conv[0] * conv[2] * channels
        channels = conv[2]
    return num


def calc_memory_count(initial_shape, convs):
    initial_size = initial_shape[0]
    num = np.prod(initial_shape)
    size = initial_size
    for conv in convs:
        size = int(np.ceil(size / conv[1]))
        num += size * conv[2]
    return num
