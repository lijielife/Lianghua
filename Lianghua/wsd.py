# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from WindPy import *
import pandas as pd
import numpy as np
import datetime
w.start()

symbols=["000001.sz","000002.SZ","000004.SZ","000005.SZ","000006.SZ","000007.SZ","000008.SZ","000009.SZ","000010.SZ","000011.sz"]

def get_stock_wsd(symbol,n_years):
    #symbols=["000001.sz","000002.SZ","000003.SZ","000004.SZ","000005.SZ","000006.SZ","000007.SZ","000008.SZ","000009.SZ","000010.SZ"]
    now=datetime.datetime.now()
    delta = datetime.timedelta(days=n_years*365)
    prev=now-delta
    symbol_data=w.wsd(symbol, "open,high,low,close,pct_chg", prev, now, "")

    fm=pd.DataFrame(symbol_data.Data,index=symbol_data.Fields,columns=symbol_data.Times)
    fm=fm.T

    fm['key']=range(0,fm.iloc[:,0].size)
    fm.dropna()#如果一行的数据中存在na就删除这一行
   # print(fm)
    return fm


#print(pinganstock)
#get_stock({"000001.SZ"},{"open,high"},{"2017-01-01"},{"2017-07-16"},"")
#for symbol in symbols:
#    get_stock(symbol)
#    print(get_stock(symbol))
print(get_stock_wsd(['000001.sz'],n_years=2))
