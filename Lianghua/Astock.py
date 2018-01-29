# coding=utf-8

from WindPy import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.finance as mpf
from datetime import datetime

import wsd
import wsi

w.start()

symbols=["000001.sz","000002.SZ","000004.SZ","000005.SZ","000006.SZ","000007.SZ","000008.SZ","000009.SZ","000010.SZ","000011.sz"]

pingan_df=wsd.get_stock_wsd('000001.sz',n_years=2)
#pingan_df.tail()

"""
#pingan_df.info()
a=pingan_df.loc['2016-07-23':'2016-07-31','OPEN']
#print(a)
b=pingan_df[np.abs(pingan_df.OPEN)>11]
#print(b)
c=pingan_df.sort_values(by='OPEN',ascending=False)[:5]
#print(c)
d=pingan_df.OPEN[:5]
#print(d)
e=pingan_df.OPEN.pct_change() #pct_change对序列从第二项开始做减法然后再除以前一项
#print(e)
f=np.round(e*100,2)
#print(f)
g=pingan_df['positive']=np.where(pingan_df.OPEN>13,1,0)
#print(pingan_df)
"""
def plot_demo(axs=None,just_series=False):
    drawer=plt if axs is None else axs
    drawer.plot(pingan_df.CLOSE,c='r')
    if not just_series:
        drawer.plot(pingan_df.CLOSE.index,pingan_df.CLOSE.values+1,c='g')
        drawer.plot(pingan_df.CLOSE.index.tolist(),(pingan_df.CLOSE.values+2).tolist(),c='b')

    plt.xlabel('time')
    plt.ylabel('close')
    plt.title('PINGAN CLOSE')
    plt.grid(True)
  #  plt.show()
def loc():
    plot_demo()
  #  fig,ax=plt.subplots(figsize=(14,8))
    plt.legend(['aaa','sss','ddd'])
    plt.show()


#plot_demo()
def plot_loc():
    _, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
    drawer = axs[0][0]
    plot_demo(drawer)
    drawer.legend(['Series', 'Numpy', 'List'], loc=0)
    # plot_demo(drawer)
    drawer = axs[0][1]
    plot_demo(drawer)
    drawer.legend(['Series', 'Numpy', 'List'], loc=1)
    drawer = axs[1][0]
    plot_demo(drawer)
    drawer.legend(['Series', 'Numpy', 'List'], loc=2)
    drawer = axs[1][1]
    plot_demo(drawer)
    drawer.legend(['Series', 'Numpy', 'List'], bbox_to_anchor=(1.05, 1),
                  loc=2,
                  borderaxespad=0.)
    plt.show()

#K线图的绘制
def candle_draw():
    __colorup__ = "red"
    __colordown__ = "green"
    pingan_part_df = pingan_df[:100]
    fig, ax = plt.subplots(figsize=(14, 7))
    qutotes = []
    for index, (d, o, c, h, l) in enumerate(
            zip(pingan_part_df.index, pingan_part_df.OPEN, pingan_part_df.CLOSE, pingan_part_df.HIGH,
                pingan_part_df.LOW)):
        d = mpf.date2num(d)
        # 日期、开盘、收盘、最高、最低组成一个tuple对象val
        val = (d, o, c, h, l)
        qutotes.append(val)
    mpf.candlestick_ochl(ax, qutotes, width=0.6, colorup=__colorup__, colordown=__colordown__)
    ax.autoscale_view()
    ax.xaxis_date()
    plt.grid(True)
    plt.show()

#pingan_df['key']=pingan_df.apply()
#print(pingan_df)
"""""
#统计涨跌幅来验证“涨势”是否正确，并将结果用各种图表展现出来
low_to_high_df=pingan_df.iloc[pingan_df[(pingan_df.CLOSE>pingan_df.OPEN)&(pingan_df.key!=pingan_df.shape[0]-1)].key.values+1]
change_ceil_floor=np.where(low_to_high_df['PCT_CHG']>0,np.ceil(low_to_high_df['PCT_CHG']),np.floor(low_to_high_df['PCT_CHG']))
change_ceil_floor=pd.Series(change_ceil_floor)
print("下跌的跌幅整数和："+str(change_ceil_floor[change_ceil_floor<0].sum()))
print("上涨的涨幅取整和："+str(change_ceil_floor[change_ceil_floor>0].sum()))

fig,axs=plt.subplots(nrows=2,ncols=2,figsize=(12,10))
change_ceil_floor.value_counts().plot(kind='bar',ax=axs[0][0])
change_ceil_floor.value_counts().plot(kind='barh',ax=axs[0][1])
change_ceil_floor.value_counts().plot(kind='kde',ax=axs[1][0])
change_ceil_floor.value_counts().plot(kind='pie',ax=axs[1][1])
plt.show()
"""""

#import seaborn as sns
#sns.distplot(pingan_df['PCT_CHG'],bins=80)
#plt.show()
#sns.jointplot(pingan_df['HIGH'],pingan_df['LOW'])
#plt.show()

pingan_close=pingan_df.CLOSE
x=np.arange(0,pingan_close.shape[0])
y=pingan_close.values
import statsmodels.api as sm
from statsmodels import regression
def regress_y(y):
    y=y
    x=np.arange(0,len(y))
    x=sm.add_constant(x)
    model=regression.linear_model.OLS(y,x).fit()
    return model
model=regress_y(y)
b=model.params[0]
k=model.params[1]
y_fit=k*x+b
#plt.plot(x,y)
#plt.plot(x,y_fit,'r')
#model.summary()
#plt.show()
import itertools
from sklearn import metrics
_,axs=plt.subplots(nrows=3,ncols=3,figsize=(15,15))
axs_list=list(itertools.chain.from_iterable(axs))
print("it is:"% axs_list)
poly=np.arange(1,10,1)
for p_cnt,ax in zip(poly,axs_list):
    p=np.polynomial.Chebyshev.fit(x,y,p_cnt)
    y_fit=p(x)
    mse=metrics.mean_squared_error(y,y_fit)
    ax.set_title('{}poly MSE={}'.format(p_cnt,mse))

    ax.plot(x,y,'',x,y_fit,'r.')
plt.show()




