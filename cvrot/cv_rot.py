

from typing import Sequence, Tuple

import math

import matplotlib.pyplot as plt
import numpy as np

import cv2
from skimage.metrics import structural_similarity

from utils import show_im



def equal_percent(im1: np.ndarray, im2: np.ndarray):
    assert im1.shape == im2.shape, f"{im1.shape}   vs   {im2.shape}"

    return (im1 != im2).sum() / im1.size



def get_image_corners(img):
    x, y = img.shape[:2]
    return np.float32([
        [0,0],
        [x-1, 0],
        [0, y-1],
        [x-1, y-1]
    ])


def transform_points(points: Sequence[Tuple[float, float]], T:np.ndarray):

    coords = cv2.transform(np.float32([ [[x, y]] for x, y in points]), T)

    return [(x, y) for x, y, z in coords[:,0,:]]







#@profile
def find_transformation_cv(img_arr_from, img_arr_to):

    ench_image = cv2.cvtColor(img_arr_to, cv2.COLOR_BGR2GRAY)
    orig_image_full = cv2.cvtColor(img_arr_from, cv2.COLOR_BGR2GRAY)

    if orig_image_full.size / ench_image.size < 3:
        resize_tf = np.identity(3)

        orig_image = orig_image_full
    else:

        print(f"warning, strong sizes difference: {orig_image_full.size / ench_image.size}")

        # тут идея в том, чтобы при сжатии более-менее оставить исходные пропорции
        target_size_img = ench_image if ench_image.shape[0] > ench_image.shape[1] and orig_image_full.shape[0] > orig_image_full.shape[1] else ench_image.T

        resize_tf = cv2.getPerspectiveTransform(
            get_image_corners(orig_image_full),
            get_image_corners(target_size_img)
        )

        orig_image = cv2.warpPerspective(
            orig_image_full, resize_tf, (target_size_img.shape[1], target_size_img.shape[0]),
            cv2.INTER_LINEAR, borderValue=(255, 255, 255))




    # if less then 10 points matched -> not the same images or higly distorted
    MIN_MATCH_COUNT = 10
    good = []

    # orig_image = np.where(np.all([img_arr_from[:, :, i] == 255 * v for i, v in enumerate((1, 0, 0))], axis=0), 0, 255).astype(np.uint8)
    # ench_image = np.where(np.all([same_color(img_arr_to[:, :, i], 255*v) for i, v in enumerate((1, 0, 0))], axis = 0), 0, 255).astype(np.uint8)


    for surf in (cv2.ORB_create(nfeatures=1500), cv2.SIFT_create()):

        try:
            #surf = cv2.ORB_create(nfeatures=1500)  #SIFT_create()
            kp1, des1 = surf.detectAndCompute(ench_image, None)
            kp2, des2 = surf.detectAndCompute(orig_image, None)
        except cv2.error as e:
            raise e

        bf = cv2.BFMatcher() #FlannBasedMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good) > MIN_MATCH_COUNT:
            break
    else:
        print("CANNOT FIND GOOD MATCHES!!!")

        print('not enough matches')

        #import winsound
        #winsound.PlaySound("SystemExit", winsound.SND_ALIAS)

        #breakpoint()
        return orig_image_full, np.identity(3)


    src_pts = np.float32([kp1[m.queryIdx].pt for m in good
                          ]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good
                          ]).reshape(-1, 1, 2)


    # # kp1_matched = ([kp1[m.queryIdx] for m in good])
    # # kp2_matched = ([kp2[m.trainIdx] for m in good])
    #
    # matches = cv2.drawMatches(ench_image, kp1, orig_image, kp2, good, None, flags=2)
    #
    # plt.figure(figsize=(20, 10))
    # plt.axis('off')
    # plt.imshow(matches), plt.show()




    # Finds a perspective transformation between two planes.
    T, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # ss = T[0, 1]
    # sc = T[0, 0]
    # scaleRecovered = math.sqrt(ss * ss + sc * sc)
    # thetaRecovered = math.atan2(ss, sc) * 180 / math.pi
    # print("""Calculated scale difference: %.2f\n
    #         Calculated rotation difference: %.2f""" % (scaleRecovered, thetaRecovered))


    # print('start matrix:')
    # print(T)
    # print()

    Tinv = np.linalg.inv(T).dot(resize_tf)
    # print('transform matrix')
    # print(Tinv)
    # print()

    # angle = np.arcsin(Tinv[0, 1]) * 360 / 2 / np.pi
    # print(f"angle: {angle}")


    rotated_old = cv2.warpPerspective(img_arr_from, Tinv, (img_arr_to.shape[1], img_arr_to.shape[0]), cv2.INTER_LINEAR, borderValue=(255, 255, 255))


    print(f"equal percent = {round(equal_percent(rotated_old, img_arr_to) * 100)}")


    return rotated_old, Tinv


