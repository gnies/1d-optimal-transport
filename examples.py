import ot_1d 
import numpy as np
import ot
import time 
import matplotlib.pyplot as plt

def test_dual():
    np.random.seed(0)
     
    # choosing an arbitrary cost function
    c = lambda x, y : - x * y # this one has the nice, becouse then c-convexity is equivalent to convexity
    # c = lambda x, y : (x-y)**2

    n = 100
    m = 80 
    x = np.random.normal(size=n)
    y = np.random.normal(size=m)
    cmat = c(x.reshape(-1, 1), y.reshape(1, -1))
    
    # creating random probability vectors
    a = np.random.random(size=n)
    a = a / np.sum(a)
    b = np.random.random(size=m)
    b = b / np.sum(b)
    
    # solving ot problem
    t0 = time.time()
    sol = ot_1d.dual(x, y, c, a=a, b=b)
    t1 = time.time()
    ot_cost = sol[0]
    f = sol[1]
    fc = sol[2]
    res_pot = ot.emd2(a, b, cmat, numItermax=1000000)
    print("OT cost (dual): ", ot_cost, "  OT cost computed with pot: ", res_pot)
    print("Time to compute solution = {:.4f}".format(t1 - t0))
    plt.title("Dual solution of the problem and its c-transform")
    sx = np.argsort(x)
    plt.plot(x[sx], f[sx], label="$f$")
    sy = np.argsort(y)
    plt.plot(y[sy], fc[sy], label="$f^c$")
    plt.legend()
    plt.show()

def test_primal():
    np.random.seed(0)
     
    # choosing an arbitrary cost function
    c = lambda x, y : - x * y # this one has the nice, becouse then c-convexity is equivalent to convexity
    # c = lambda x, y : (x-y)**2

    n = 100
    m = 80 
    x = np.random.normal(size=n)
    y = np.random.normal(size=m)
    cmat = c(x.reshape(-1, 1), y.reshape(1, -1))
    
    # creating random probability vectors
    a = np.random.random(size=n)
    a = a / np.sum(a)
    b = np.random.random(size=m)
    b = b / np.sum(b)
    
    # solving ot problem
    t0 = time.time()
    sol = ot_1d.primal(x, y, c, a=a, b=b)
    t1 = time.time()
    ot_cost = sol[0]
    pi = sol[1]
    res_pot = ot.emd2(a, b, cmat, numItermax=1000000)
    print("OT cost (primal): ", ot_cost, "  OT cost computed with pot: ", res_pot)
    print("Time to compute solution = {:.4f}".format(t1 - t0))
    # good iteractive plot of a geodesic might fit nicely here 

# quick test
if __name__=="__main__":
    test_dual()
    test_primal()
