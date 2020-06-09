# -*- coding: utf-8 -*-
"""
Created on Sat May 30 23:03:00 2020

@author: qtckp
"""


coins = [2, 3, 7]
change = 12

def coin_change(score, string = '', deep=0):
    print(str(score)+' '+string +' from coin '+ str(deep))
    if score <= 0:
        return 0
    else:
        for i in coins:
            if score >= i:
                coin_change(score - i,string +'1', i)

coin_change(change)
