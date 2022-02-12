
from typing import Optional, List, Tuple

import numpy as np
from numba import guvectorize, jit


class Kernel:
    def __init__(self, arr: np.ndarray):

        x, y = np.nonzero(arr)
        coords: List[Tuple[int, int]] = [(i, j) for i, j in zip(x, y)]
        values: List[int] = [arr[v] for v in coords]
        
        self.values = values
        self.coords = coords

        self.info = sorted([(int(val), x, y) for val, (x, y) in zip(values, coords)], key = lambda v: v[0])

        self.positive = [(x, y) for val, x, y in self.info if val == 1]
        self.negative = [(x, y) for val, x, y in self.info if val == -1]
        self.not_one = [(val, x, y) for val, x, y in self.info if abs(val) != 1]
        # на 0 не проверяю, так как это условие по определению выполняется

    @property
    def is_binary(self):
        return len(self.negative) == 0 and len(self.not_one) == 0

    @property
    def x(self):
        return np.array([t[1] for t in self.info], dtype = np.uint8)
    @property
    def y(self):
        return np.array([t[2] for t in self.info], dtype = np.uint8)
    @property
    def v(self):
        return np.array([t[0] for t in self.info], dtype = np.int8)


def conv_apply(
        image: np.ndarray,
        kernel: np.ndarray,
        maxon: Optional[float] = None
) -> np.ndarray:

    return conv_numba_maxon(image, kernel, maxon)

    xs, ys = image.shape

    xpad = int((kernel.shape[0] - 1)/2)
    ypad = int((kernel.shape[1] - 1)/2)

    ker = Kernel(kernel[::-1, ::-1])

    image2 = np.zeros(
        (xs + 2*xpad, ys+ 2*ypad),
        dtype=image.dtype
    )
    image2[xpad:(xs + xpad), ypad:(ys + ypad)] = image

    t = np.zeros(
        (xs, ys),
        dtype=np.int32 if image.dtype != np.float else np.float
    )
    
    for x, y in ker.positive:
        t += image2[x:(xs + x), y:(ys + y)]
    
    if not ker.is_binary:  # есть ещё слагаемые, которые можно уже добавить; этот код самый долгий,
        # лучше не кидать во внутренние функции (наверное)
        for x, y in ker.negative:
            t -= image2[x:(xs + x), y:(ys + y)]
        
        for val, x, y in ker.not_one:
            t += val*image2[x:(xs + x), y:(ys + y)]

    return t if maxon is None else t >= maxon



@guvectorize(['void(uint8[:, :], uint8[:], uint8[:], int8[:], uint8, uint8[:, :], uint8[:, :])'],
             '(big_rows,big_cols),(k),(k),(k),(),(rows,cols)->(rows,cols)',
             nopython=True, target='cpu'
             )
def _conv_maxon(values_array, kernel_x, kernel_y, kernel_val, maxon, dummy, target_array):

    xs, ys = target_array.shape
    ker_len = kernel_x.size

    for row in range(ker_len):
        kx = kernel_x[row]
        ky = kernel_y[row]
        value = kernel_val[row]

        for i in range(xs):
            for j in range(ys):
                if values_array[i + kx, j + ky] != 0:
                    target_array[i, j] += value

    # for i in range(xs):
    #     for j in range(ys):
    #
    #         counter = 0
    #         for row in range(ker_len):
    #             if values_array[i + kernel_x[row], j + kernel_y[row]] != 0:
    #                 counter += kernel_val[row]
    #                 if counter >= maxon:
    #                     target_array[i, j] = 1
    #                     break



@jit('uint8[:, :](uint8[:, :], int8[:, :], uint8, uint16, uint16)',
             nopython=True,
             )
def _conv_maxon_jit(values_array, kernel_array, maxon,  xs, ys):

    target_array = np.zeros((xs, ys), dtype = np.uint8)
    ker_len = kernel_array.shape[0]

    for i in range(xs):
        for j in range(ys):

            counter = 0
            for row in range(ker_len):
                xyv = kernel_array[row]
                if values_array[i + xyv[0], j + xyv[1]] > 0:
                    counter += xyv[2]
                    if counter >= maxon:
                        target_array[i, j] = 1
                        #break

            # if counter < maxon:
            #     target_array[i, j] = 0

    return target_array



def conv_numba_maxon(image: np.ndarray, kernel: np.ndarray, maxon: int):

    assert np.issubdtype(image.dtype, bool) and kernel.dtype==np.int8, "wrong types of arrays"

    xs, ys = image.shape
    xpad = int((kernel.shape[0] - 1)/2)
    ypad = int((kernel.shape[1] - 1)/2)
    ker = Kernel(kernel[::-1, ::-1])

    image2 = np.zeros(
        (xs + 2*xpad, ys+ 2*ypad),
        dtype=np.uint8
    )
    image2[xpad:(xs + xpad), ypad:(ys + ypad)] = image

    result = np.zeros(
        (xs, ys),
        dtype=np.uint8
    )

    _conv_maxon(image2, ker.x, ker.y, ker.v, maxon, result, result)

    #from dreamocr.cython_packs.main import conv_module

    import dreamocr.cython_packs.conv as cc


    return result.astype(bool)



