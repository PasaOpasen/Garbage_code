
import cv2
import numpy as np
import copy
import pylab
import time
import sys
import sklearn.neighbors
import scipy.optimize

import matplotlib.pyplot as plt


def icp(a, b,
        max_time=4
        ):

    def res(p, src, dst):
        T = np.matrix([[np.cos(p[2]), -np.sin(p[2]), p[0]],
                          [np.sin(p[2]), np.cos(p[2]), p[1]],
                          [0, 0, 1]])
        n = np.size(src, 0)
        xt = np.ones([n, 3])
        xt[:, :-1] = src
        xt = (xt * T.T).A
        d = np.zeros(np.shape(src))
        d[:, 0] = xt[:, 0] - dst[:, 0]
        d[:, 1] = xt[:, 1] - dst[:, 1]
        r = np.sum(np.square(d[:, 0]) + np.square(d[:, 1]))
        return r

    def jac(p, src, dst):
        T = np.matrix([[np.cos(p[2]), -np.sin(p[2]), p[0]],
                          [np.sin(p[2]), np.cos(p[2]), p[1]],
                          [0, 0, 1]])
        n = np.size(src, 0)
        xt = np.ones([n, 3])
        xt[:, :-1] = src
        xt = (xt * T.T).A
        d = np.zeros(np.shape(src))
        d[:, 0] = xt[:, 0] - dst[:, 0]
        d[:, 1] = xt[:, 1] - dst[:, 1]
        dUdth_R = np.matrix([[-np.sin(p[2]), -np.cos(p[2])],
                                [np.cos(p[2]), -np.sin(p[2])]])
        dUdth = (src * dUdth_R.T).A
        g = np.array([np.sum(2 * d[:, 0]),
                         np.sum(2 * d[:, 1]),
                         np.sum(2 * (d[:, 0] * dUdth[:, 0] + d[:, 1] * dUdth[:, 1]))])
        return g

    def hess(p, src, dst):
        n = np.size(src, 0)
        T = np.matrix([[np.cos(p[2]), -np.sin(p[2]), p[0]],
                          [np.sin(p[2]), np.cos(p[2]), p[1]],
                          [0, 0, 1]])
        n = np.size(src, 0)
        xt = np.ones([n, 3])
        xt[:, :-1] = src
        xt = (xt * T.T).A
        d = np.zeros(np.shape(src))
        d[:, 0] = xt[:, 0] - dst[:, 0]
        d[:, 1] = xt[:, 1] - dst[:, 1]
        dUdth_R = np.matrix([[-np.sin(p[2]), -np.cos(p[2])], [np.cos(p[2]), -np.sin(p[2])]])
        dUdth = (src * dUdth_R.T).A
        H = np.zeros([3, 3])
        H[0, 0] = n * 2
        H[0, 2] = np.sum(2 * dUdth[:, 0])
        H[1, 1] = n * 2
        H[1, 2] = np.sum(2 * dUdth[:, 1])
        H[2, 0] = H[0, 2]
        H[2, 1] = H[1, 2]
        d2Ud2th_R = np.matrix([[-np.cos(p[2]), np.sin(p[2])], [-np.sin(p[2]), -np.cos(p[2])]])
        d2Ud2th = (src * d2Ud2th_R.T).A
        H[2, 2] = np.sum(2 * (
                    np.square(dUdth[:, 0]) + np.square(dUdth[:, 1]) + d[:, 0] * d2Ud2th[:, 0] + d[:, 0] * d2Ud2th[
                                                                                                                :, 0]))
        return H

    t0 = time.time()
    init_pose = (0, 0, 0)
    src = np.array([a.T], copy=True).astype(np.float32)
    dst = np.array([b.T], copy=True).astype(np.float32)
    Tr = np.array([[np.cos(init_pose[2]), -np.sin(init_pose[2]), init_pose[0]],
                      [np.sin(init_pose[2]), np.cos(init_pose[2]), init_pose[1]],
                      [0, 0, 1]])

    # print("src", np.shape(src))
    # print("Tr[0:2]", np.shape(Tr[0:2]))

    src = cv2.transform(src, Tr[0:2])
    p_opt = np.array(init_pose)
    T_opt = np.array([])
    error_max = sys.maxsize
    first = False
    while not (first and time.time() - t0 > max_time):
        distances, indices = sklearn.neighbors.NearestNeighbors(n_neighbors=1, algorithm='auto', p=3).fit(
            dst[0]).kneighbors(src[0])
        p = scipy.optimize.minimize(res, [0, 0, 0], args=(src[0], dst[0, indices.T][0]), method='Newton-CG', jac=jac,
                                    hess=hess).x
        T = np.array([[np.cos(p[2]), -np.sin(p[2]), p[0]], [np.sin(p[2]), np.cos(p[2]), p[1]]])
        p_opt[:2] = (p_opt[:2] * np.matrix(T[:2, :2]).T).A
        p_opt[0] += p[0]
        p_opt[1] += p[1]
        p_opt[2] += p[2]
        src = cv2.transform(src, T)
        Tr = (np.matrix(np.vstack((T, [0, 0, 1]))) * np.matrix(Tr)).A
        error = res([0, 0, 0], src[0], dst[0, indices.T][0])

        if error < error_max:
            error_max = error
            first = True
            T_opt = Tr

    p_opt[2] = p_opt[2] % (2 * np.pi)

    return T_opt, error_max



def draw_result(T, error):
    dx = T[0, 2]
    dy = T[1, 2]
    rotation = np.arcsin(T[0, 1]) * 360 / 2 / np.pi

    print("T", T)
    print("error", error)
    print("rotation°", rotation)
    print("dx", dx)
    print("dy", dy)

    result = cv2.transform(np.array([data.T], copy=True).astype(np.float32), T).T
    plt.plot(template[0], template[1], label="template")
    plt.plot(data[0], data[1], label="data")
    plt.plot(result[0], result[1], label="result: " + str(rotation) + "° - " + str([dx, dy]))
    plt.legend(loc="upper left")
    plt.axis('square')
    plt.show()






if __name__ == '__main__':
    import random

    n1 = 1500
    n2 = 1000
    bruit = 1 / 10
    center = [random.random() * (2 - 1) * 3, random.random() * (2 - 1) * 3]
    radius = random.random()
    deformation = 2

    template = np.array([
        [np.cos(i * 2 * np.pi / n1) * radius * deformation for i in range(n1)],
        [np.sin(i * 2 * np.pi / n1) * radius for i in range(n1)]
    ])

    data = np.array([
        [np.cos(i * 2 * np.pi / n2) * radius * (1 + random.random() * bruit) + center[0] for i in range(n2)],
        [np.sin(i * 2 * np.pi / n2) * radius * deformation * (1 + random.random() * bruit) + center[1] for i in
         range(n2)]
    ])

    T, error = icp(data, template)

    draw_result(T, error)



















