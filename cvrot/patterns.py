
import os

from tqdm import tqdm

import numpy as np

import PIL
import cv2

from utils import show_im, show_nonzero

from ICP import icp



# cross = np.array([
#     [0,0,1,0,0],
#     [0,0,1,0,0],
#     [1,1,1,1,1],
#     [0,0,1,0,0],
#     [0,0,1,0,0]
# ])


cross = np.ones((10,20), dtype = np.uint8)


# def unit_circle_vectorized_filled(r):
#     A = np.arange(-r,r+1)**2
#     dists = np.sqrt(A[:,None] + A)
#     return ((dists-r)<0.5).astype(int)
#
#
# cross = unit_circle_vectorized_filled(6)



# colors = [
#     (1, 0, 0),
#     (0, 1, 0),
#     (0, 0, 1),
#     (1, 1, 0),
#     (1, 0, 1),
#     (0, 1, 1)
# ]

colors = [(1,0,0)]#*10


def color_pattern(pattern):

    arrs = [ np.array([255 - 255*pattern * (1 - v) for v in col]) for col in colors]

    return [np.swapaxes(np.swapaxes(r, 0, 1), 1, 2) for r in arrs ]




def draw_pattern(img_arr, pattern = cross, count = 5):

    colored_pattern = color_pattern(pattern)[0]

    px, py = colored_pattern.shape[:2]
    ix, iy = img_arr.shape[:2]

    x, y = np.nonzero(np.sum(img_arr, axis = -1) == 0)

    # if len(x) > 0:
    #     xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    # else:
    if True:
        xmin = ymin = 0
        xmax = ix-1
        ymax = iy-1

    def find_shift(min_shift, pattern_size):
        return max(0, int((min_shift-pattern_size)/2))

    xmin = find_shift(xmin, px)
    xmax = ix - find_shift(ix-xmax, px)
    ymin = find_shift(ymin, py)
    ymax = iy - find_shift(iy-ymax, py)


    y_shift = int((ymax - ymin - count*py)/(count-1))
    y_shift_small = int(((ymax - ymin)/2 - count * py) / (count - 1))

    y_index = ymin
    y_index_small = ymin
    for _ in range(count):
        for xslice, yslice in (
                (slice(xmin, xmin + px), slice(y_index, y_index + py)),
                (slice(xmax - px, xmax), slice(y_index_small, y_index_small + py)),
        ):
            img_arr[xslice, yslice, :] = np.minimum(img_arr[xslice, yslice, :], colored_pattern)

        y_index += py + y_shift
        y_index_small += py + y_shift_small


    return img_arr



# def same_color(arr, value):
#     return np.abs(arr - value) < 5
def same_color(arr, value):
    return value - arr < 6 if value == 255 else arr < 15


def find_rotation(img_arr_from, img_arr_to):

    #pattern_axes = [np.arange(3)[np.array(col, dtype = bool)] for col in colors]

    Ts = []

    for pat in colors:#pattern_axes:

        xf, yf = np.nonzero(np.all([img_arr_from[:,:,i] == 255*v for i, v in enumerate(pat)], axis = 0))
        xt, yt = np.nonzero(np.all([same_color(img_arr_to[:, :, i], 255*v) for i, v in enumerate(pat)], axis = 0))

        print(f"len found = {len(xt)}")

        if len(xf) > 4 and len(xt) > 4:

            T, error = icp(
                np.array([
                    xt, yt
                ]),
                np.array([
                    xf, yf
                ])
            )

            print(T)
            print(f"error = {error}")
            print(f"angle: {np.arcsin(T[0, 1]) * 360 / 2 / np.pi}")

            Ts.append(T)

    T_avg = np.mean(Ts, axis = 0)
    angle = np.arcsin(T_avg[0, 1]) * 360 / 2 / np.pi
    print(f"mean angle: {angle}")

    rotated_old = cv2.warpPerspective(img_arr_from, T_avg, (img_arr_from.shape[1], img_arr_from.shape[0]), cv2.INTER_LINEAR, borderValue=(255, 255, 255))


    return angle

































if __name__ == '__main__':

    t = color_pattern(cross)


    all_white = np.full((100,100,3), 255)

    filled = draw_pattern(all_white.copy())

    pass





















