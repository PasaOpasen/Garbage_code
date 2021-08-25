

import os

from collections import Counter

import numpy as np

import cv2


def get_image_counter(img):
    return Counter(img.ravel())


def pre_process_image(img, save_in_file = None, min_color = 0, max_color = 255):

    pre = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ct = get_image_counter(pre)

    # Правило такое:
    # самый частый цвет -- это фон, его мы сразу делаем белым для большего контраста (иногда он серый)
    # и всё, что белее фона -- становится белым
    #
    counts = np.zeros(256)
    for key, val in ct.items():
        counts[key] = val
    fore_color = counts.argmax()
    border_white = min(max_color, fore_color)

    pre[pre >= border_white] = 255

    ct = {key: val for key, val in ct.items() if min_color < key < border_white}


    min_count = 0# int(pre.size * 0.002)
    max_count = int(pre.size * 0.005)


    for key, val in ct.items():
        pre[pre == key] = 0 if min_count <= val <= max_count else 255


    if save_in_file is not None:
        cv2.imwrite(save_in_file, pre)

    return pre



if __name__ == "__main__":

    number = 3

    in_file = f"{number}.jpg"
    pre_file = os.path.join("data2", f"{number}_pre.png")

    img = cv2.imread(in_file)

    pre_processed = pre_process_image(img, pre_file)


    # cv2.imwrite(out_file, vis)