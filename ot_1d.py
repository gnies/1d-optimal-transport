import numpy as np

def dual(x, y, c, a=None, b=None, ordered=False, rightmargin=1e-13):
    """
    Computes optimal transport on the real line for arbitrary convex cost in near linear time. In particular we find the c-convex dual solution f and concave function g = f^c that solve

    .. math::
        arg\max_{f} <f^c,b> - <f,a>

    where :

    - f^c denotes the c-transform of f
    - a and b are source and target weights 
    - c is the cost function
    - x and y are the support points of the sourse and target measures.

    Parameters
    ----------
    x : array_like
        support points of the first measure
    y : array_like
        support points of the second measure
    c : cost function
        function with respect to which the c-convex conjugate is evaluated.
        It needs to support bradcasting of numpy arrays.
    a : array_like
        probability simplex of the first measure, if none then the uniform distribution on x is assumed.
    b : array_like
        probability simplex of the second measure, if none then the uniform distribution on y is assumed.
    ordered : bool
        if true then it is assumed that x and y are already sorted arrays.
        Under this assumption some computations can be saved
    rightmargin : float
        when comuting the quantile function of the measure b a small margin has to be added to enforce right
        continuity

    Returns
    ot_cost : float
           the optimal transport cost
    f : ndarray
           the dual solution of the problem. The function f will be c-convex on the support of a
    g : ndarray
           the other dual solution of the problem. The function g will be c-concave on the support of b and satisfy g = f^c.
    """
    n = len(x)
    m = len(y)
    if a is None:
        a = np.ones(n) / n
    if b is None:
        b = np.ones(m) / m

    if ordered == False:
        # If the support points are not ordered then we sort things out
        i_x = np.argsort(x)
        x = x[i_x]
        a_perm = a[i_x]
        i_y = np.argsort(y)
        y = y[i_y]
        b_perm = b[i_y]
    
    #  cumulative distirbutions
    ca = np.cumsum(a) 
    cb = np.cumsum(b) 

    # bins need some small tollerance to the right to avoid numerical rounding errors
    bins = cb + rightmargin  

    # right=True becouse quantile function is right continuous.
    index_y_star = np.digitize(ca, bins, right=True)  
    try: 
        y_star = y[index_y_star]
    except IndexError:
        raise Exception("Problem infeasible. Check that a and b are in the simplex")

    # Dual solution f
    f0 = 0  # value of dual solution in x_0 is now fixed to zero.
    df = c(x[:-1], y_star[:-1]) - c(x[1:], y_star[:-1])
    f1n = np.cumsum(df)
    f = np.hstack([f0, f1n])

    # Dual solution g, which is the c-transform of f
    index_g = np.digitize(y, y_star, right=True)  
    index_g[np.where(index_g == len(x))] = len(x)-1
    g = f[index_g] + c(x[index_g], y)

    otcost = np.sum(g*b) - np.sum(f*a)

    if ordered == False:
        # if support points where not sorted already then we bring back f to original order
        inv_p_x = np.argsort(i_x)
        inv_p_y = np.argsort(i_y)
        f = f[inv_p_x]
        g = g[inv_p_y]
    return otcost, f, g

def primal(x, y, c, a=None, b=None, ordered=False, rightmargin=1e-13):
    """
    Computes the optimal transport plan between two arbirary discrete measures on R.
    Parameters
    ----------
    x : array_like
        support points of the first measure
    y : array_like
        support points of the second measure
    c : cost function
        function with respect to which the c-convex conjugate is evaluated.
        It needs to support bradcasting of numpy arrays.
    a : array_like
        probability simplex of the first measure, if none then the uniform distribution on x is assumed.
    b : array_like
        probability simplex of the second measure, if none then the uniform distribution on y is assumed.
    rightmargin : float
        when comuting the quantile function of the measure b a small margin has to be added to enforce right
        continuity

    Returns
    -------
    ot_cost : float
        the optimal transport cost
    pi : float
        the optimal transport plan
    """
    n = len(x)
    m = len(y)
    if a is None:
        a = np.ones(n) / n
    if b is None:
        b = np.ones(m) / m

    if ordered == False:
        # If the support points are not ordered then we sort things out
        i_x = np.argsort(x)
        x = x[i_x]
        a_perm = a[i_x]
        i_y = np.argsort(y)
        y = y[i_y]
        b_perm = b[i_y]

    #  cumulative distirbutions
    ca = np.cumsum(a)
    cb = np.cumsum(b)

    # points on which we need to evaluate the quantile functions
    cba = np.sort(np.hstack([ca, cb]))

    # construction of first quantile function index
    bins = ca + rightmargin # bins need some small tollerance to avoid numerical rounding errors
    index_qx = np.digitize(cba, bins, right=True)    # right=True becouse quantile function is 
                                                     # right continuous
    # quantile function would now be given by qx = x[index_qx]

    # construction of second quantile function index
    bins = cb + rightmargin
    index_qy = np.digitize(cba, bins, right=True)    # right=True becouse quantile function is 
                                                     # right continuous

    # weights for the inegral
    h = np.diff(np.hstack([0, cba]))
    pi = np.zeros((len(x), len(y)))
    hn0 = np.where(h >= rightmargin)
    try:
        pi[index_qx[hn0], index_qy[hn0]] = h[hn0]
    except IndexError:
        raise Exception("Problem infeasible. Check that a and b are in the simplex")
    ot_cost = np.sum(c(x.reshape((-1, 1)), y.reshape((1, -1))) *  pi)

    if ordered == False:
        # if support points where not sorted already then we bring back f to original order
        inv_p_x = np.argsort(i_x)
        inv_p_y = np.argsort(i_y)
        pi = pi[inv_p_x.reshape(-1, 1), inv_p_y.reshape(1, -1)]
    return ot_cost, pi
