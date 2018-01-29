# -*- encoding:utf-8 -*-
"""
    全局环境配置模块
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import re
import platform
import sys
import warnings
from enum import Enum
from os import path

from CKFixs import six

import numpy as np
import pandas as pd

#暂时仅支持windows和mac
ck_is_mac_os=platform.system().lower().find("windows")<0 and sys.platform != "win32"
#判断python版本环境
ck_is_py3 = six.PY3
ck_is_ipython = True
#判断是否在主进程
ck_main_pid=os.getpid()

try:

    __IPYTHON__
except NameError:
    ck_is_ipython=False

try:
    import psutil
    cpu_cnt=psutil.cpu_count(logical=True)
except ImportError:
    if ck_is_py3:
        cpu_cnt=os.cpu_count()
    else:
        import multiprocessing as mp

        cpu_cnt=mp.cpu_count()

except:
    cpu_cnt=4

#pandas忽略赋值警告
pd.options.mode.chained_assignment=None

#numpy,pandas显示控制，默认开启
display_control=True
if display_control:
    #Dataframe表格最大显示行数
    pd.options.display.max_rows=20
    #最大显示列数
    pd.options.display.max_columns=20
    #pandas精度浮点数显示4位
    pd.options.display.precisions=4
    #浮点数显示4位，不使用科学技术法
    np.set_printoptions(precision=4,suppress=True)

#是否忽略所有警告，默认关闭
ignore_all_warning=False
#默认忽略所有库警告
ignore_all_lib_warining=True
if ignore_all_lib_warining:
    try:
        import matplotlib as mpl

        mpl.warnings.filterwarnings('ignore')
        mpl.warnings.simplefilter('ignore')

        import sklearn

        sklearn.warnings.filterwarnings('ignore')
        sklearn.warnings.simplefilter('ignore')
    except:
        pass

if ignore_all_warning:
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')


#**********************数据目录************************


def str_is_cn(a_str):

    def to_unicode(text,encoding=None,errors='strict'):

        if isinstance(text,six.text_type):
            return text
        if not isinstance(text,(bytes,six.text_type)):
            raise TypeError('to_unicode must receive a bytes,str or unicode'
                            'object,got %s' % type(text).__name__)
        if encoding is None:
            encoding = 'utf-8'
        try:
            decode_text=text.decode(encoding,errors)
        except:
            decode_text=text.decode('gbk' if encoding=='utf-8' else 'utf-8',errors)
        return decode_text

    cn_re=re.compile(u'[\u4e00-\u9fa5]+')
    try:
        is_cn_path=cn_re.search(to_unicode(a_str)) is not None
    except:
        is_cn_path=True
    return is_cn_path








