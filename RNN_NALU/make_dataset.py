import numpy as np
import torch


def wave_generate(scale=20, length=1000, sample_num=100, random_shift=5, func=np.sin):
    x = np.empty((sample_num, length), 'float64')
    x[:] = np.array(range(length)) + np.random.randint(-1 * random_shift, random_shift, sample_num).reshape(
        sample_num, 1)
    data = func(x / scale)
    torch.save(data, 'data.torch')


if __name__ == '__main__':
    wave_generate()
