import math
import numpy as np

def arcosh(x=None):
    assert x is not None, "x参数未定义！"
    if not isinstance(x,complex) and np.abs(x)>=1:
        if x + np.sqrt(x*x-1) < 0:
            return complex(np.log(abs(x + np.sqrt(x*x-1))),np.pi)
        else:
            return math.acosh(x)
    else:
        return np.log(x+sqrt(np.power(x,2)-1))

def arrange(start=None,end=None,step=None):
    assert start is not None, "start参数未定义！"
    assert end is not None,   "end参数未定义！"
    assert step is not None,  "step参数未定义！"
    assert step != 0, "step参数定义错误！"
    res = []
    if step > 0:
        while start <= end:
            res.append(start)
            start = start + step
    else:
        while start >= end:
            res.append(start)
            start = start + step
    return np.resize(a=np.array(res),new_shape=[1,len(res)])

def cosh(x):
    if not isinstance(x, complex):
        return math.cosh(x)
    else:
        return (np.exp(x)+np.exp(-1*x))/2

def nextpow2(x):
    return np.ceil(np.log2(x))

def sqrt(x):
    if not isinstance(x, complex):
        if x >= 0:
            return np.sqrt(x)
        else:
            return complex(0,sqrt(np.abs(x)))
    else:
        return np.sqrt(x)
