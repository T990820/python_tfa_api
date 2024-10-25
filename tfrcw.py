def tfrcw(x=None,t=None,N=None,g=None,h=None,sigma=None,trace=None):
    if x is None and t is None and N is None and g is None and h is None and sigma is None and trace:
        raise ValueError("At least 1 parameter required")
    xrow = x.shape[0]
    xcol = x.shape[1]
    if xcol == 0 or xcol > 2:
        raise ValueError("X must have one or two columns")