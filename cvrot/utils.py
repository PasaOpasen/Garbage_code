

import numpy as np

from PIL import Image



def show_im(img_arr):
    Image.fromarray(img_arr.astype(np.uint8)).show()


def show_nonzero(x, y, img_orig):

    im = np.full_like(img_orig, 255)

    im[(x, y)] = 0

    show_im(im)







