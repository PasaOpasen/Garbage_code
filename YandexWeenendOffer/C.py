# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 20:14:06 2021

@author: qtckp
"""


import numpy as np

n = int(input().rstrip())
arr = np.empty((n,3), dtype = np.int64)

for i in range(n):
    arr[i, :] = np.fromstring(input(), count = 3,  dtype = int, sep = ' ')

  
flag = False

u, count = np.unique(arr[:,2], return_counts=True)



for c in np.unique(u)[::-1][:-1]:
    
    color_mask = arr[:,2] == c
    
    arr_color = arr[color_mask,:2]
    arr_outer = arr[~color_mask,:]
    
    amin = arr_color[:,0].max()
    bmin = arr_outer[:,0].max()
    amax = arr_color[:,1].min()
    bmax = arr_outer[:,1].min()
    
    if not ((arr_color[:,0].max() < arr_outer[:,0].min() and arr_color[:,1].min() > arr_outer[:,1].max()) or (arr_outer[:,0].max() < arr_color[:,0].min() and arr_outer[:,1].min() > arr_color[:,1].max())):
        for i in range(arr_color.shape[0]):
            l, r = tuple(arr_color[i,:])
            
            l_r = arr_outer[:,0] < l
            r_r = arr_outer[:,1] > r
            
            #l_l = arr_outer[:,0] > l
            #r_l = arr_outer[:,1] < r
            
            if np.any(np.logical_xor(l_r , r_r)):
                print('NO')
                flag = True
                break
    if flag:
        break
    
    arr = arr_outer
    
if not flag:
    print('YES')
        


