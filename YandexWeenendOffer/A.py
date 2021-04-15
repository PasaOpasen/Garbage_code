# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 19:30:27 2021

@author: qtckp
"""


import itertools



all_carts = list(range(2, 15)) * 4

sums = [sum(t) for t in itertools.combinations(all_carts, 6)]

