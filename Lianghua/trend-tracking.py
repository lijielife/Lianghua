# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from WindPy import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

import wsd
import wsi
w.start()

#消除复制警告
pd.options.mode.chained_assignment=None

#选定pingan两年的股票走势数据
kl_pd=wsd.get_stock_wsd('000001.sz',n_years=2)

#定义N1，N2
N1=20
N2=10

#加入新的数据列，代表N1天内的最高价格
kl_pd['n1_high']=pd.Series(kl_pd['HIGH']).rolling(window=N1).max()#新版的写法
#填充n1_high前N1行数据
expan_max=pd.Series(kl_pd['CLOSE']).expanding().max()#(kl_pd['CLOSE'])
kl_pd['n1_high'].fillna(value=expan_max,inplace=True)

#用类似方法构建N2天内最低价格
kl_pd['n2_low']=pd.Series(kl_pd['LOW']).rolling(window=N2).min()#(kl_pd['LOW'],window=N2)
expan_min=pd.Series(kl_pd['CLOSE']).expanding().min()#(kl_pd['CLOSE'])
kl_pd['n2_low'].fillna(value=expan_min,inplace=True)

#构建signal列
buy_index=kl_pd[kl_pd['CLOSE']>kl_pd['n1_high'].shift(1)].index
kl_pd.loc[buy_index,'signal']=1

sell_index=kl_pd[kl_pd['CLOSE']<kl_pd['n2_low'].shift(1)].index
kl_pd.loc[sell_index,'signal']=0

#将操作信号转化为持股信号
kl_pd['keep']=kl_pd['signal'].shift(1)
kl_pd['keep'].fillna(method='ffill',inplace=True)

#计算基准收益
kl_pd['benchmark_profit']=np.log(kl_pd['CLOSE']/kl_pd['CLOSE'].shift(1))
#计算趋势突破收益
kl_pd['trend_profit']=kl_pd['keep']*kl_pd['benchmark_profit']
#可视化收益情况对比
kl_pd[['benchmark_profit','trend_profit']].cumsum().plot(grid=True,figsize=(14,7))

plt.show()
