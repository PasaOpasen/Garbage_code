
from typing import Callable, Sequence

import os, sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import shutil

from tqdm import tqdm
import imagehash
import cv2

sys.path.append('../../../')

from backend.doca.compare_imgs import compare_images, binarized, matches_arrs, get_sim_func

from dreamocr.utils.files import mkdir
from dreamocr.utils.rectangles import bboxes_union_volume
from dreamocr.preprocessing.tablecells.utils import draw_cells




def image_hashes(img_gray: np.ndarray, rows: int = 25, cols: int = 25):

    img = Image.fromarray(img_gray).convert('L')

    r = img.size[1]*2/25
    c = img.size[0]*2/25
    r = c = max(int(min(r, c)) + 1, 10)

    hashes = {}
    for i in range(0, img.size[1] - r, r//2):
        for j in range(0, img.size[0] - c, c//2):
            box = (i, j, i + r, j + c)
            crop = img.crop(box = box)
            hashes[box] = imagehash.phash(crop)

    print(img.size, r, len(hashes))

    return hashes

def comp_images_hashes(img1: np.ndarray, img2: np.ndarray, thresh: float = 0.97):

    im1, im2 = img1, img2
    if im1.size < im2.size:
        im1, im2 = im2, im1
    s = min(im2.shape[:2]) / min(im1.shape[:2]) * 1.1
    if s < 1:
        new_size = tuple([int(s*ss) for ss in im1.shape[:2]])
        im1 = cv2.resize(im1, new_size, interpolation=cv2.INTER_LINEAR)

    h1 = image_hashes(im1)
    h2 = image_hashes(im2)

    mat = np.empty((len(h1), len(h2)))

    for i, hash1 in enumerate(h1.values()):
        sqr = len(hash1.hash)**2
        mat[i] = np.array([hash1-hash2 for hash2 in h2.values()]) / sqr


    mat = 1 - mat

    mat[mat<thresh] = 0
    x, y = np.nonzero(mat)
    scores = [((x_, y_), mat[x_, y_]) for x_, y_ in zip(x, y)]
    scores.sort(key = lambda p: (p[-1], -abs(p[0][0] - p[0][1])))
    print(f"{x.size} points")

    filtered = []
    while scores:
        filtered.append(scores[-1])
        x_, y_ = scores[-1][0]
        scores = [((xx, yy), v) for (xx, yy), v in scores if xx!=x_ and yy!=y_]

    print(f"{len(filtered)} selected")

    if len(filtered) == 0:
        return 0

    boxes1_inds = set([xx for (xx, _), _ in filtered])
    boxes2_inds = set([yy for (_, yy), _ in filtered])

    im1_boxes = [b for i, b in enumerate(h1.keys()) if i in boxes1_inds]
    im2_boxes = [b for i, b in enumerate(h2.keys()) if i in boxes2_inds]

    vol_1 = bboxes_union_volume(im1_boxes) / (im1.shape[0] * im1.shape[1])
    vol_2 = bboxes_union_volume(im2_boxes) / (im2.shape[0] * im2.shape[1])
    print('---->', vol_1, vol_2)

    return min(vol_1, vol_2)



from geneticalgorithm2 import geneticalgorithm2 as ga
from geneticalgorithm2 import AlgorithmParams


PATH_GA = './ga_phash_100_stud'
mkdir(PATH_GA)
counter = 0
hash_func = imagehash.phash
def images_hash(img1, img2):
    a = hash_func(img1)
    b = hash_func(img2)
    return (a - b) / len(a.hash) ** 2



def plot_areas_on_images(img1, box1, img2, box2, path: str):

    im1 = Image.fromarray(np.dstack((np.asarray(img1),) * 3))
    im2 = Image.fromarray(np.dstack((np.asarray(img2),) * 3))

    score = images_hash(img1, img2)
    im1 = draw_cells(im1, [box1], width=2)
    im2 = draw_cells(im2, [box2], width=2)

    fig = plt.figure(figsize=(8, 8))

    grid = plt.GridSpec(2, 1, hspace=0.2, wspace=0.1)
    subs = (
        fig.add_subplot(grid[0, 0], xticklabels=[], yticklabels=[]),
        fig.add_subplot(grid[1, 0], xticklabels=[], yticklabels=[])
    )

    for sub, im in zip(subs, (im1, im2)):
        sub.imshow(np.asarray(im))
        sub.set_axis_off()

    fig.suptitle(f"hash similarity = {1 - score:0.2%}")

    plt.savefig(path, dpi = 200)
    plt.close()



def sim_ga(im1: np.ndarray, im2: np.ndarray):
    global counter

    MIN_OK = 0.04
    MAX_ITERS = 5

    img1 = Image.fromarray(im1).convert('L')
    img2 = Image.fromarray(im2).convert('L')


    t1 = (0, 0, img1.size[0], img1.size[1])
    t2 = (0, 0, img2.size[0], img2.size[1])

    bound1 = (
        (0, t1[2]),
        (0, t1[3]),
        (0, t1[2]),
        (0, t1[3])
    )
    bound2 = (
        (0, t2[2]),
        (0, t2[3]),
        (0, t2[2]),
        (0, t2[3])
    )


    cache1, cache2 = {}, {}

    def hash_box(box, img, cache):
        tp = tuple(box)
        hash = cache.get(tp)
        if hash is None:
            hash = hash_func(img.crop(tp))
            cache[tp] = hash
        return hash

    def make_minimized_function(img, cache, const_hash):
        def minimized_f(box):
            if box[0] > box[2]:
                box[2], box[0] = box[0], box[2]
            elif box[0] == box[2]:
                return 1
            if box[1] > box[3]:
                box[3], box[1] = box[1], box[3]
            elif box[1] == box[3]:
                return 1

            hash = hash_box(box, img, cache)
            return (hash - const_hash) / len(hash.hash) ** 2
        return minimized_f


    best_res = 1
    reversed = False
    k = 0
    while k < MAX_ITERS:

        const_hash = hash_box(t2, img2, cache2)
        minimized_f = make_minimized_function(img1, cache1, const_hash)

        if k == 0 and not reversed:
            print(f"\nstart score = {minimized_f(t1)}")
            assert images_hash(img1, img2) == minimized_f(t1)

        bounds = bound1
        def generate_gen(size = 25):
            arrs = [np.random.randint(a, b, size) for a, b in bounds]
            return np.array(
                [
                    np.minimum(arrs[0], arrs[2]),
                    np.minimum(arrs[1], arrs[3]),
                    np.maximum(arrs[0], arrs[2]),
                    np.maximum(arrs[1], arrs[3])
                ]
            ).T


        model = ga(
            function=minimized_f,
            dimension = 4,
            variable_type='int',
            variable_boundaries=bounds,
            algorithm_parameters=AlgorithmParams(
                max_iteration_without_improv=60
            )
        )

        start_gen = generate_gen(100)
        start_gen[0] = np.array(t1)
        if best_res < 1:
            assert minimized_f(start_gen[0]) == best_res

        model.run(
            start_generation=start_gen,
            stop_when_reached=MIN_OK,

            studEA=True,

            no_plot=True,
            disable_printing=True,
            disable_progress_bar=True,

            seed = 1
        )

        t1, new_result = tuple(model.best_variable), model.best_function

        if (new_result == best_res and (k > 1 or reversed)) or new_result <= MIN_OK:
            best_res = new_result
            print(k, best_res)
            print('exit cuz no progress')
            break

        assert new_result <= best_res
        best_res = new_result

        if reversed:
            k += 1

        print(f"dicts sizes: {len(cache1)}, {len(cache2)}")

        # swap
        print(t1, t2, best_res, reversed)
        t1, t2 = t2, t1
        bound1, bound2 = bound2, bound1
        cache1, cache2 = cache2, cache1
        img1, img2 = img2, img1
        reversed = not reversed



    # if reversed:
    #     t1, t2 = t2, t1

    min_volume = min(
        (t1[2] - t1[0]) *  (t1[3] - t1[1]) / (img1.height * img1.width),
        (t2[2] - t2[0]) * (t2[3] - t2[1]) / (img2.height * img2.width),
    )

    plot_areas_on_images(
        img1, t1,
        img2, t2,
        path = os.path.join(PATH_GA, f"{counter + 1}.png")
    )
    counter += 1

    return min_volume * (1 - best_res)
















def get_sim_matrix(images, metric: Callable[[np.ndarray, np.ndarray], float]):
    matrix = np.empty((len(images), len(images)))
    for i, img1 in tqdm(enumerate(images)):
        for j, img2 in enumerate(images[i:]):
            sim = metric(img1, img2)
            x, y = i, i + j
            matrix[x, y] = sim
            matrix[y, x] = sim
    return matrix


def plot_sim_matrix(images: Sequence[np.ndarray], matrix: np.ndarray, path: str):
    fig = plt.figure(figsize=(20, 20))

    count = len(images)

    grid = plt.GridSpec(count + 1, count + 1, hspace=0.1, wspace=0.1)

    for i in range(count):

        subs = (
            fig.add_subplot(grid[0, i + 1]),
            fig.add_subplot(grid[i + 1, 0])
        )

        for sub in subs:
            sub.imshow(images[i])
            sub.set_axis_off()

    sub = fig.add_subplot(grid[1:, 1:])
    sub.imshow(matrix)
    sub.set_axis_off()

    max_val = matrix.max()
    # print(matrix)

    if 0 < max_val <= 1:
        mt = np.round(matrix, 2)
    elif max_val < 100:
        mt = np.round(matrix)
    else:
        mt = np.round(matrix / max_val, 2)

    # Loop over data dimensions and create text annotations.
    for i in range(count):
        for j in range(count):
            text = sub.text(j, i, mt[i, j],
                            ha="center", va="center", color="w")

    plt.savefig(path, dpi = 300)
    plt.close()




PATH_IMGS = '../../../../dreamocr/debug_inputs/labels'

names = [os.path.join(PATH_IMGS, f) for f in os.listdir(PATH_IMGS)]
names = sorted([n for n in names if os.path.isfile(n)])

images = [Image.open(n) for n in names]
arrs = [binarized(np.asarray(img.convert('L'))) for img in images]


matrix = get_sim_matrix(
    arrs,
    metric = sim_ga #comp_images_hashes #lambda a, b: compare_images(a, b, 'sift', resize=True)
)
np.savetxt('mat.txt', matrix, delimiter=' ')
plot_sim_matrix(arrs, matrix, os.path.join(PATH_GA, 'images_sim.png'))


# matches_dir = './matches'
# shutil.rmtree(matches_dir, ignore_errors=True)
# Path(matches_dir).mkdir(exist_ok=True, parents=True)
#
# for arr, name in matches_arrs:
#
#     f, *other = name.split()
#
#     dr = os.path.join(matches_dir, f)
#     Path(dr).mkdir(exist_ok=True, parents=True)
#
#     plt.figure(figsize=(10, 5))
#     plt.axis('off')
#     plt.imshow(arr)
#     plt.savefig(os.path.join(dr, f"{' '.join(other)}.png"))
#     plt.close()














