# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 22:03:13 2021

@author: qtckp
"""

n = int(input().rstrip())

dct = {
       1:[],
       2:[],
       3:[],
       4:[],
       }


for _ in range(n):
    x, y, a, b = [int(v) for v in input().rstrip().split()]
    
    if a == 1:
        if b == 1:
            c = 1
        else:
            c = 4
    else:
        if b == 1:
            c = 2
        else:
            c = 3

    dct[c].append((x,y))
    


if any((len(val) == 0 for val in dct.values())):
    
    print('NO')

else:
    
    flag = False
    
    for x, y in dct[1]:
        
        p2 = [(x_,y_) for x_,y_ in dct[2] if y_ >= y]
        if len(p2) == 0: continue
    
        p3 = [(x_,y_) for x_,y_ in dct[3] if y_ >= y and x_ >= x]
        if len(p3) == 0: continue
    
        p4 = [(x_,y_) for x_,y_ in dct[4] if x_ >= x]
        if len(p4) == 0: continue
        
        for x2, y2 in p2:
            
            p23 =  [(x_,y_) for x_,y_ in p3 if x_ >= x2]
            if len(p23) == 0: continue
        
            p24 = [(x_,y_) for x_,y_ in p4 if y_ <= y2 and x_ >= x2]
            if len(p24) == 0: continue 
            
            for x3, y3 in p23:
                
                p234 =  [(x_,y_) for x_,y_ in p24 if y_ <= y3]
                
                if len(p234) == 0:
                    continue 
                else:
                    flag = True
                    break
            if flag:
                break
        if flag:
            print('YES')
            break
    
    if not flag:
        print('NO')
        
    