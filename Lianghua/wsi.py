# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from WindPy import *
import numpy as np
import pandas as pd
import datetime

w.start()

def get_stock_wsi(symbol,n_days):

    now=datetime.datetime.now()
    delta=datetime.timedelta(days=n_days)
    prev=now-delta

    wsi_data=w.wsi(symbol,"open,low,high,close,pct_chg",prev,now,"")

    fm=pd.DataFrame(wsi_data.Data,index=wsi_data.Fields,columns=wsi_data.Times)
    fm=fm.T

    fm['key'] = range(0, fm.iloc[:, 0].size)
    return fm

#a=get_stock_wsi(['000001.sz'],30)
#print(a)