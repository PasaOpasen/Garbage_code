# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 19:38:14 2021

@author: qtckp
"""

import numpy as np
import pandas as pd

df = pd.read_csv('log.csv')




sess = df.query('date >=  "2020-04-19" & date < "2020-04-20"')

total = 0

for user in pd.unique(sess.user):
    tmp = sess.query("user == @user")
    
    dt = pd.to_datetime(tmp.date.str.replace('_', ' '))
    counter = 1
    for i in range(1, len(dt)):
        if (dt.iloc[i] - dt.iloc[i-1]).total_seconds()/60 >= 30:
            counter += 1
    
    total += counter
    


sec = df.query("event_type == 2 & parameter == 'video'")

sec['date'] = pd.to_datetime(sec.date.str.replace('_', ' ')).dt.date

un = sec.loc[:, ['date', 'user']].drop_duplicates()

un.groupby('date').count().max()




ev = pd.to_datetime(df.date.str.replace('_', ' '))

mx = 0
d = None
delta = pd.Timedelta(5, unit="m")
for i in range(ev.shape[0]):
    date_s = ev.iloc[i]
    date_end = date_s + delta
    count = (ev.iloc[i:] < date_end).sum()
    if count >= mx:
        mx = count
        d = date_s
        print(f"{mx} {d}")

    


