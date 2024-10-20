import math
import scipy
import numpy as np

import utils

def tftb_window(N=None,name=None,param=None,param2=None):
    h = None
    assert N is not None, "at least 1 parameter is required"
    assert N > 0, "N should be strictly positive."
    if name is None:
        name = 'Hamming'
    name = name.upper()
    if name == "RECTANG" or name == "RECT":
        h = np.ones(shape=[N,1])
    elif name == "HAMMING":
        h = np.array(range(1,N+1)).T
        h = 0.54 - 0.46*np.cos(2.0*np.pi*h/(N+1))
        h = np.resize(a=h,new_shape=[N,1])
    elif name == "HANNING" or name == "HANN":
        h = np.array(range(1, N + 1)).T
        h = 0.50 - 0.50 * np.cos(2.0 * np.pi * h/(N+1))
        h = np.resize(a=h, new_shape=[N, 1])
    elif name == "KAISER":
        beta = None
        if N is not None and name is not None and param is not None and param2 is None:
            beta=param
        else:
            beta = 3.0 * np.pi
        ind = utils.arrange(start=-(N-1)/2,end=(N-1)/2,step=1).T
        ind = ind * 2 / N
        beta = 3 * np.pi
        h = scipy.special.jv(0,0+beta*np.sqrt(1.0-np.power(ind,2))*complex(0,1)) / scipy.special.jv(0,complex(0,1)*beta).real
    elif name == "NUTTALL":
        ind = utils.arrange(start=-(N-1)/2,end=(N-1)/2,step=1).T*2.0*np.pi/N
        h = 0.3635819+0.4891775*np.cos(ind) +0.1363995*np.cos(2.0*ind) +0.0106411*np.cos(3.0*ind)
    elif name == "BLACKMAN":
        ind = utils.arrange(start=-(N-1)/2,end=(N-1)/2,step=1).T*2.0*np.pi/N
        h = 0.42 + 0.50 * np.cos(ind) + 0.08 * np.cos(2.0 * ind)
    elif name == "HARRIS":
        ind = utils.arrange(start=1, end=N, step=1).T*2.0*np.pi/(N+1)
        h = +0.35875-0.48829 * np.cos(ind)+0.14128 * np.cos(2.0 * ind)-0.01168 * np.cos(3.0 * ind)
    elif name == "BARTLETT" or name == "TRIANG":
        h = np.concatenate((utils.arrange(start=1,end=N,step=1),utils.arrange(start=N,end=1,step=-1)),0)
        h = np.resize(np.array(2*np.min(h,axis=0)/(N+1)),new_shape=[N,1])
    elif name == "BARTHANN":
        one_N = utils.arrange(start=1, end=N, step=1)
        min_one_N_one = np.resize(a=np.min(np.concatenate((one_N,utils.arrange(end=1, start=N, step=-1)),0),0),new_shape=[1,N])
        h = 0.38 * (1.0 - np.cos(2.0 * np.pi * one_N / (N + 1)).T) + 0.48 * min_one_N_one.T/(N+1)
    elif name == "PAPOULIS":
        ind = utils.arrange(start=1, end=N, step=1).T*np.pi/(N+1)
        h=np.sin(ind)
    elif name == "GAUSS":
        if N is not None and name is not None and param is not None and param2 is None:
            K = param
        else:
            K = 0.005
        h = np.exp(np.log(K)*np.power(np.resize(a=np.linspace(start=-1,stop=1,num=N),new_shape=[1,N]),2)).T
    elif name == "PARZEN":
        ind = np.abs(utils.arrange(start=-(N-1)/2,end=(N-1)/2,step=1)).T*2/N
        temp = 2*np.power(1.0-ind,3)
        h = temp-np.power((1-2.0*ind),3)
        h = np.expand_dims(np.min(np.concatenate((temp,h),1),1),1)
    elif name == "HANNA":
        if N is not None and name is not None and param is not None and param2 is None:
            L = param
        else:
            L = 1
        ind = utils.arrange(start=0,end=N-1,step=1).T
        h=np.power(np.sin((2*ind+1)*np.pi/(2*N)),2*L)
    elif name == "DOLPH" or name == "DOLF":
        if N % 2 == 0:
            oddN = 1
            N = 2 * N + 1
        else:
            oddN = 0
        if N is not None and name is not None and param is not None and param2 is None:
            A = np.power(10,param/20)
        else:
            A = 0.001
        K = N - 1
        Z0 = math.cosh(math.acosh(1.0 / A) / K)
        x0 = math.acos(1 / Z0) / np.pi
        x = utils.arrange(start=0,end=K,step=1)/N
        # indices1=find((x<x0)|(x>1-x0))
        indices1 = []
        for row, row_vector in enumerate(x):
            for col, num in enumerate(row_vector):
                if num<x0 or num>1-x0:
                    indices1.append([row,col])
        # indices2=find((x>=x0)&(x<=1-x0))
        indices2 = []
        for row, row_vector in enumerate(x):
            for col, num in enumerate(row_vector):
                if x0 <= num <= 1-x0:
                    indices2.append([row,col])
        # h(indices1)= cosh(K*acosh(Z0*cos(pi*x(indices1))))
        h = np.empty(shape=[1, N],dtype=complex)
        for _, indice in enumerate(indices1):
            h[indice[0],indice[1]] = utils.cosh(K*utils.arcosh(Z0*math.cos(np.pi*x[indice[0],indice[1]])))
        # h(indices2)= cos(K*acos(Z0*cos(pi*x(indices2))))
        for _, indice in enumerate(indices2):
            h[indice[0],indice[1]] = utils.cosh(K*utils.arcosh(Z0*math.cos(np.pi*x[indice[0],indice[1]])))
        # h=fftshift(real(ifft(A*real(h))));h=h'/h(K/2+1)
        h = np.fft.fftshift(np.fft.ifft(A*h.real).real)
        h = h.T/h[0,int(K/2)]
        if oddN == 1:
            tmp = []
            for i in range(K):
                if i % 2 == 1:
                    tmp.append(h[i,0])
        h = np.expand_dims(np.array(tmp),1)
    elif name == "NUTBESS":
        beta = None
        if N is not None and name is not None and param is not None and param2 is None:
            beta = param
            nu = 0.5
        elif N is not None and name is not None and param is not None and param2 is not None:
            beta = param
            nu = param2
        else:
            beta = 3 * np.pi
            nu = 0.5
        ind = np.resize(a=utils.arrange(start=-(N-1)/2,end=(N-1)/2,step=1),new_shape=[N,1])*2/N
        h = np.multiply(np.power(np.sqrt(1-np.power(ind,2)),nu),scipy.special.jv(nu,complex(0,1)*beta*np.sqrt(1-np.power(ind,2))).real/scipy.special.jv(nu,beta*complex(0,1)).real)
    elif name == "SPLINE":
        assert param is not None, "Three or four parameters required for spline windows"
        if N is not None and name is not None and param is not None and param2 is None:
            nfreq = param
            p = np.pi * N * nfreq / 10.0
        if N is not None and name is not None and param is not None and param2 is not None:
            nfreq = param
            p = param2
        ind = np.resize(a=utils.arrange(start=-(N - 1) / 2, end=(N - 1) / 2, step=1), new_shape=[N, 1])
        h = np.power(np.sinc((0.5*nfreq/p)*ind),p)
    elif name == "FLATTOP":
        ind = np.resize(a=utils.arrange(start=-(N - 1) / 2, end=(N - 1) / 2, step=1), new_shape=[N, 1]) * 2 * np.pi / (N - 1)
        h = +0.2810639+0.5208972 * np.cos(ind)+0.1980399 * np.cos(2.0 * ind)
    else:
        raise ValueError("unknown window name")
    return h