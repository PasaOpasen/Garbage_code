# -*- coding: utf-8 -*-
"""
Created on Sat May 30 02:07:19 2020

@author: qtckp
"""

import numpy as np

a= [1, 2, 3, 2.5]

b= [[0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]]

print(np.dot(b,a))
print(np.dot(a,b))

