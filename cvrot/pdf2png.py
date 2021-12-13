
import os
import time

from tqdm import tqdm

from pathlib import Path

import numpy as np
import pandas as pd

from pdf2image import convert_from_path
from skimage.metrics import structural_similarity

import PIL
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from skimage.segmentation import flood_fill

import cv2
import math
from scipy import ndimage


from patterns import draw_pattern, find_rotation

from cv_rot import find_transformation_cv


white = (255,255,255)



def isascii(s):
    """Check if the characters in string s are in ASCII, U+0-U+7F."""
    return len(s) == len(s.encode())

def load_img(path):

    if isascii(path):
        return cv2.imread(path)[..., ::-1]
    return np.asarray(Image.open(path))













def create_rotated_images(dir = './dataset2_pdf/'):

    # for index in range(1, 12):
    #     pages = [np.array(pg) for pg in convert_from_path(f"{index}.pdf")]

    for file in os.listdir(dir):

        name = os.path.basename(file).split('.')[0]

        pages = [np.array(pg) for pg in convert_from_path(os.path.join(dir, file) )]


        for page, im in enumerate(pages):

            Im = Image.fromarray(im)
            Im.save(f"./dataset2/{name}_{page}.png")

            for angle in (10, 20, 70, 80, 90, 100, 110, 160, 170, 180, 190, 200, 250, 260, 270, 280, 290, 340, 350, 360):

                Image.fromarray(im).rotate(angle, PIL.Image.NEAREST, expand = 1, fillcolor = white).save(f"./rotated/{name}_{page}_{angle}.png")

                # loc_im = Image.fromarray(im).rotate(angle, PIL.Image.NEAREST, expand=1, fillcolor=white).convert('L')
                #
                # loc_arr = np.asarray(loc_im)
                #
                # xb, yb = loc_arr.shape
                #
                # for x, y in (
                #         (0, 0),
                #         (xb-1, 0),
                #         (0, yb-1),
                #         (xb-1, yb-1)
                # ):
                #     flood_fill(loc_arr, (x,y), 255, in_place=True)
                #
                # Image.fromarray(loc_arr).save(f"./rotated/{index}_{page}_{angle}.png")


def reader_pdf_to_img():

    for file in tqdm(os.listdir('./rotated_pdf/')):

        arr = np.array(convert_from_path(os.path.join('./rotated_pdf/', file))[0])

        arr = np.asarray(
            Image.fromarray(arr)#.convert('L')
        )
        #
        # xb, yb = arr.shape
        #
        # for x, y in (
        #         (0, 0),
        #         (xb - 1, 0),
        #         (0, yb - 1),
        #         (xb - 1, yb - 1)
        # ):
        #     flood_fill(arr, (x, y), 0, in_place=True)



        Image.fromarray(arr).save(f"./after_reader/{os.path.basename(file).split('.')[0]}.png")



def plot_patterns_on_images(dir = './rotated/'):

    for file in tqdm(os.listdir(dir)):

        arr = cv2.imread(os.path.join(dir, file))[...,::-1]

        new_arr = draw_pattern(arr)

        Image.fromarray(new_arr).save(f"./rotated_crosses/{os.path.basename(file).split('.')[0]}.png")


def check_angles_icp(dir = './after_reader/'):

    target = []
    predict = []

    for file in os.listdir(dir):

        before = load_img(os.path.join('./rotated/', file))
        after = load_img(os.path.join(dir, file))

        t = time.process_time()
        new_im, Mat = find_transformation_cv(before, after) #find_rotation(before, after)
        print('TIME = ', time.process_time() - t)

        current_angle = int(os.path.basename(file).split('.')[0].split('_')[-1])


        try:

            base = os.path.basename(file).split('.')[0]
            base_path = Path(os.path.join('./output/', base))
            base_path.mkdir(exist_ok=True)

            Image.fromarray(before).save(os.path.join(str(base_path), base + '.png'))
            Image.fromarray(after).save(os.path.join(str(base_path), base + '_reader.png'))
            Image.fromarray(new_im).save(os.path.join(str(base_path), base + '_reconstucted.png'))


            angle = np.arcsin(Mat[0, 1]) * 360 / 2 / np.pi
            #print(f"was {current_angle}, found {round(angle)}")

            target.append(current_angle)
            predict.append(angle)

        except Exception as e:
            print(str(e))

    target = np.array(target)
    predict = np.array(predict)

    df = pd.DataFrame({
        'real': target,
        'found': predict
    }).sort_values('real')

    #df['shift'] = [round(v) % 90 for v in df['found']]

    df['shift'] = [90 - round(v) % 90 for v in df['found']]

    pass



def check_angles_icp2(dir = './reader_splitted/'):

    used_dirs = set([p.split(' ')[-1] for p in os.listdir("./reader_splitted/output")]) if os.path.exists("./reader_splitted/output") else set()


    sims = []
    times = []
    fails_count = 0

    for file in tqdm(os.listdir(os.path.join(dir, 'before_reader'))):
        base = os.path.basename(file).split('.')[0]

        if base in used_dirs:
            continue

        print(f"basename: {base}")

        before = load_img(os.path.join(dir,'before_reader', file))
        after = load_img(os.path.join(dir,'after_reader', file))

        t = time.process_time()
        new_im, Mat = find_transformation_cv(before, after) #find_rotation(before, after)
        t = time.process_time() - t
        print('TIME = ', t )
        times.append(t)

        if (Mat - np.identity(3)).sum() == 0:
            fails_count += 1

            base_path = Path( os.path.join('./reader_splitted/bad_examples/', base))
            base_path.mkdir(parents=True, exist_ok=True)
            Image.fromarray(before).save(os.path.join(str(base_path), base + '.png'))
            Image.fromarray(after).save(os.path.join(str(base_path), base + '_reader.png'))

            continue

        sim = structural_similarity(new_im, after, multichannel=True)
        print(f"similarity = {sim}", '\n')
        sims.append(sim)

        #current_angle = int(os.path.basename(file).split('.')[0].split('_')[-1])
        try:

            M = np.abs(Mat)
            def minmax_condition(a, b, k = 5):
                return max(a, b)/ min(a, b) > k

            if max(M[0,0], M[1,1], M[2, 2]) > 100 or minmax_condition(M[0,0], M[1,1]):
                base_path = Path( os.path.join('./reader_splitted/bad_matrix/', str(round(100 * sim, 3)).replace('.', '_') + ' ' + base))
            else:
                #base_path = Path(os.path.join('./reader_splitted/output/', base))
                base_path = Path(os.path.join('./reader_splitted/output/', str(round(100*sim, 3)).replace('.', '_') + ' ' + base))


            base_path.mkdir(parents=True, exist_ok=True)

            Image.fromarray(before).save(os.path.join(str(base_path), base + '.png'))
            Image.fromarray(after).save(os.path.join(str(base_path), base + '_reader.png'))
            Image.fromarray(new_im).save(os.path.join(str(base_path), base + '_reconstucted.png'))

            import pandas as pd
            pd.DataFrame(Mat, columns=['x', 'y', 'z']).to_csv(os.path.join(str(base_path), base + '_mat.csv'), index = False)


            #angle = np.arcsin(Mat[0, 1]) * 360 / 2 / np.pi
            #print(f"was {current_angle}, found {round(angle)}")


        except Exception as e:
            print(str(e))


    pass






def find_angle_cv(file):

    img_before = cv2.imread(file)

    #cv2.imshow("Before", img_before)
    #key = cv2.waitKey(0)

    img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

    angles = []

    for [[x1, y1, x2, y2]] in lines:
        #cv2.line(img_before, (x1, y1), (x2, y2), (255, 0, 0), 3)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    #cv2.imshow("Detected lines", img_before)
    #key = cv2.waitKey(0)

    median_angle = np.median(angles)
    #img_rotated = ndimage.rotate(img_before, median_angle)

    #print(f"Angle is {median_angle:.04f}")
    #cv2.imwrite('rotated.jpg', img_rotated)
    return median_angle

def check_angles(dir = './rotated/'):

    target = []
    predict = []

    for file in os.listdir(dir):

        angle = find_angle_cv(os.path.join(dir, file))
        current_angle = int(os.path.basename(file).split('.')[0].split('_')[-1])

        print(f"was {current_angle}, found {round(angle)}")

        target.append(current_angle)
        predict.append(angle)

    target = np.array(target)
    predict = np.array(predict)

    df = pd.DataFrame({
        'real': target,
        'found': predict
    }).sort_values('real')

    #df['shift'] = [round(v) % 90 for v in df['found']]

    df['shift'] = [90 - round(v) % 90 for v in df['found']]

    pass


def check_angle_from_shape():

    dir_before = './rotated/'
    dir_after = './after_reader/'

    def to_corner(y_, x_):
        return math.degrees(math.atan2(y_, x_))

    for file in os.listdir(dir_after):

        try:

            a, b = cv2.imread(os.path.join(dir_before, file)).shape[:2]
            x, y = cv2.imread(os.path.join(dir_after, file)).shape[:2]

            t = (x*x + y*y - (a*a + b*b)) / 4

            x1 = (x - math.sqrt(x * x - 4 * t))/2
            x2 = (x + math.sqrt(x * x - 4 * t)) / 2

            y1 = (y - math.sqrt(y * y - 4 * t)) / 2
            y2 = (y + math.sqrt(y * y - 4 * t)) / 2


            assert x1+x2 == x
            assert y1+y2 == y
            assert round(x2/y2, 5) == round(y1/x1, 5)
            assert round(x1*x2, 2) == t
            assert round(y1*y2, 2) == t

            print(f"{os.path.basename(file).split('.')[0].split('_')[-1]}  {to_corner(x2, y2)} {to_corner(y2, x2)}")
        except:
            pass






if __name__ == '__main__':

    #create_rotated_images()

    #plot_patterns_on_images()

    #reader_pdf_to_img()

    #check_angles('./after_reader/')

    #check_angle_from_shape()

    check_angles_icp2()


