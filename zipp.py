# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 18:40:58 2020

@author: qtckp
"""


t = ([(1,),(2,),(3,)],[('a',),('b',),('c',)])

r = {key[0]: value[0] for key, value, in zip(t[0],t[1])}








