# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from WindPy import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime



#消除复制警告
pd.options.mode.chained_assignment=None

#选定pingan两年的股票走势数据
kl_pd=wsd.get_stock_wsd('000001.sz',n_years=2)
#头一年的数据为训练数据
train_kl=kl_pd[:250]
#后一年的数据为测试数据
test_kl=kl_pd[250:]

#画出两部分数据的收盘曲线
#tmp_df=pd.DataFrame(np.array([train_kl.CLOSE.values,test_kl.CLOSE.values]).T,columns=['train','test'])
#tmp_df[['train','test']].plot(subplots=True,grid=True,figsize=(14,7))

#训练数据的收盘价格均值和标准差
close_mean=train_kl.CLOSE.mean()
close_std=train_kl.CLOSE.std()

#构造买入和卖出阈值
sell_signal=close_mean+close_std/3
buy_signal=close_mean-close_std/3

#寻找测试数据中满足买入条件的时间序列
buy_index=test_kl[test_kl['CLOSE']<=buy_signal].index
#将找到的买入时间的信号设置为1，代表买入操作
test_kl.loc[buy_index,'signal']=1

#寻找满足卖出条件的时间序列，并将信号设置为0，代表卖出
sell_index=test_kl[test_kl['CLOSE']>=sell_signal].index
test_kl.loc[sell_index,'signal']=0

#假设都是全仓操作，所以signal=keep，即1代表全部买入，0代表卖出空仓
test_kl['keep']=test_kl['signal']
test_kl['keep'].fillna(method='ffill',inplace=True)
#print(test_kl.signal)

#加入benchmark_profit计算每一天的收益；shift(1)的作用是对序列的value在index不变的情况下，向后移动股价序列一个单位
#np.log()函数计算当日收盘价相对于前一天收盘价的上涨幅度
test_kl['benchmark_profit']=np.log(test_kl['CLOSE']/test_kl['CLOSE'].shift(1))
#计算收益趋势
test_kl['trend_profit']=test_kl['keep']*test_kl['benchmark_profit']

#将基准收益和策略收益可视化对比显示
#test_kl[['benchmark_profit','trend_profit']].cumsum().plot(grid=True,figsize=(14,7))
test_kl[['benchmark_profit','trend_profit']].cumsum().apply(np.exp).plot(grid=True,figsize=(14,8))
plt.show()
