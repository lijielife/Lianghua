# -*- encoding:utf-8 -*-
from __future__ import print_function
from __future__ import division

# import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# noinspection PyUnresolvedReferences
#import abu_local_env
import abupy
from abupy import ABuSymbolPd
from WindPy import *
import wsd
w.start()

# warnings.filterwarnings('ignore')
sns.set_context(rc={'figure.figsize': (14, 7)})
# 使用沙盒数据，目的是和书中一样的数据环境
#abupy.env.enable_example_env_ipython()

#kl_pd = ABuSymbolPd.make_kl_df('usTSLA', n_folds=2)
kl_pd=wsd.get_stock_wsd('000001.sz',n_years=2)


"""
    第七章 量化系统——入门：三只小猪股票投资的故事
    abu量化系统github地址：https://github.com/bbfamily/abu (您的star是我的动力！)
    abu量化文档教程ipython notebook：https://github.com/bbfamily/abu/tree/master/abupy_lecture
"""


def sample_711():
    """
    7.1.1 趋势跟踪和均值回复的周期重叠性
    :return:
    """

    sns.set_context(rc={'figure.figsize': (14, 7)})
    sns.regplot(x=np.arange(0, kl_pd.shape[0]), y=kl_pd.CLOSE.values, marker='+')
    plt.show()

    from abupy import ABuRegUtil
    deg = ABuRegUtil.calc_regress_deg(kl_pd.CLOSE.values)
    plt.show()
    print('趋势角度:' + str(deg))

    start = 0
    # 前1/4的数据
    end = int(kl_pd.shape[0] / 4)
    # 将x也使用arange切割
    x = np.arange(start, end)
    # y根据start，end进行切片
    y = kl_pd.CLOSE.values[start:end]
    sns.regplot(x=x, y=y, marker='+')
    plt.show()

    start = int(kl_pd.shape[0] / 4)
    # 向前推1/4单位个时间
    end = start + int(kl_pd.shape[0] / 4)
    sns.regplot(x=np.arange(start, end), y=kl_pd.CLOSE.values[start:end],
                marker='+')
    plt.show()


def sample_712_1():
    """
    7.1.2 均值回复策略
    :return:
    """
    # 头一年（[:252]）作为训练数据, 美股交易中一年的交易日有252天
    train_kl = kl_pd[:245]
    # 后一年（[252:]）作为回测数据
    test_kl = kl_pd[245:]

    # 分别画出两部分数据收盘价格曲线
    tmp_df = pd.DataFrame(
        np.array([train_kl.CLOSE.values, test_kl.CLOSE.values]).T,
        columns=['train', 'test'])

    tmp_df[['train', 'test']].plot(subplots=True, grid=True,
                                   figsize=(14, 7))
    plt.show()


def sample_712_2(show=True):
    """
    7.1.2 均值回复策略
    :return:
    """
    train_kl = kl_pd[:245]
    test_kl = kl_pd[245:]

    # 训练数据的收盘价格均值
    CLOSE_mean = train_kl.CLOSE.mean()
    # 训练数据的收盘价格标准差
    CLOSE_std = train_kl.CLOSE.std()

    # 构造卖出信号阀值
    sell_signal = CLOSE_mean + CLOSE_std / 3
    # 构造买入信号阀值
    buy_signal = CLOSE_mean - CLOSE_std / 3

    # 可视化训练数据的卖出信号阀值，买入信号阀值及均值线
    if show:
        # 训练集收盘价格可视化
        train_kl.CLOSE.plot()
        # 水平线，买入信号线, lw代表线的粗度
        plt.axhline(buy_signal, color='r', lw=3)
        # 水平线，均值线
        plt.axhline(CLOSE_mean, color='black', lw=1)
        # 水平线， 卖出信号线
        plt.axhline(sell_signal, color='g', lw=3)
        plt.legend(['train CLOSE', 'buy_signal', 'CLOSE_mean', 'sell_signal'],
                   loc='best')
        plt.show()

        # 将卖出信号阀值，买入信号阀值代入回归测试数据可视化
        plt.figure(figsize=(14, 7))
        # 测试集收盘价格可视化
        test_kl.CLOSE.plot()
        # buy_signal直接代入买入信号
        plt.axhline(buy_signal, color='r', lw=3)
        # 直接代入训练集均值CLOSE
        plt.axhline(CLOSE_mean, color='black', lw=1)
        # sell_signal直接代入卖出信号
        plt.axhline(sell_signal, color='g', lw=3)
        # 按照上述绘制顺序标注
        plt.legend(['test CLOSE', 'buy_signal', 'CLOSE_mean', 'sell_signal'],
                   loc='best')
        plt.show()

        print('买入信号阀值:{} 卖出信号阀值:{}'.format(buy_signal, sell_signal))
    return train_kl, test_kl, buy_signal, sell_signal


def sample_712_3(show=True):
    """
    7.1.2 均值回复策略
    :return:
    """
    train_kl, test_kl, buy_signal, sell_signal = sample_712_2(show=False)

    # 寻找测试数据中满足买入条件的时间序列
    buy_index = test_kl[test_kl['CLOSE'] <= buy_signal].index

    # 将找到的买入时间系列的信号设置为1，代表买入操作
    test_kl.loc[buy_index, 'signal'] = 1
    # 表7-2所示
    if show:
        print('test_kl[52:57]:\n', test_kl[52:57])

    # 寻找测试数据中满足卖出条件的时间序列
    sell_index = test_kl[test_kl['CLOSE'] >= sell_signal].index

    # 将找到的卖出时间系列的信号设置为0，代表卖出操作
    test_kl.loc[sell_index, 'signal'] = 0
    # 表7-3所示
    if show:
        print('test_kl[48:53]:\n', test_kl[48:53])

    # 由于假设都是全仓操作所以signal＝keep，即1代表买入持有，0代表卖出空仓
    test_kl['keep'] = test_kl['signal']
    # 将keep列中的nan使用向下填充的方式填充，结果使keep可以代表最终的交易持股状态
    test_kl['keep'].fillna(method='ffill', inplace=True)

    # shift(1)及np.log下面会有内容详细讲解
    test_kl['benchmark_profit'] = \
        np.log(test_kl['CLOSE'] / test_kl['CLOSE'].shift(1))

    # 仅仅为了说明np.log的意义，添加了benchmark_profit2，只为对比数据是否一致
    test_kl['benchmark_profit2'] = \
        test_kl['CLOSE'] / test_kl['CLOSE'].shift(1) - 1

    if show:
        # 可视化对比两种方式计算出的profit是一致的
        test_kl[['benchmark_profit', 'benchmark_profit2']].plot(subplots=True,
                                                                grid=True,
                                                                figsize=(
                                                                    14, 7))
        plt.show()

    # test_kl['CLOSE'].shift(1): test_kl['CLOSE'] / test_kl['CLOSE'].shift(1) = 今日收盘价格序列／昨日收盘价格序列
    print('test_kl[CLOSE][:5]:\n', test_kl['CLOSE'][:5])
    print('test_kl[CLOSE].shift(1)[:5]:\n', test_kl['CLOSE'].shift(1)[:5])
    # np.log
    print('np.log(220 / 218), 220 / 218 - 1.0:', np.log(220 / 218), 220 / 218 - 1.0)

    return test_kl


def sample_712_4():
    """
    7.1.2 均值回复策略
    :return:
    """
    test_kl = sample_712_3(show=True)

    test_kl['trend_profit'] = test_kl['keep'] * test_kl['benchmark_profit']
    test_kl['trend_profit'].plot(figsize=(14, 7))
    plt.show()

    print(test_kl.signal)
    test_kl[['benchmark_profit', 'trend_profit']].cumsum().plot(grid=True,figsize=(14, 7))
    plt.show()

 #   test_kl[['benchmark_profit', 'trend_profit']].cumsum().apply(np.exp).plot(grid=True)
 #   plt.show()

sample_712_4()
#print(kl_pd)