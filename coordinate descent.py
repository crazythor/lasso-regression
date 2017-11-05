import math
import random
import numpy as np
from numpy import linalg as LA
from array import *


def gradient_descent(x_k, step_length,A,b):
    f_prime = np.dot(A.T, np.dot(A,x_k.T) - b) # derivertive of f_function 
    print 'f_prime',f_prime
    y = x_k - step_length * f_prime
    return y 

#--------------------------------------------------------------------------------
# primary function f(x) = (||Ax - b||^2) /2 
def f_func(A,x,b):
    f_value = math.pow(LA.norm(np.dot(A,x.T) - b),2) / 2
    return f_value
#soft-threshkolding function
def soft_func(b,a):
    if(a>b):
        return a-b
    elif(a<-b):
        return b+a
    else:
        return 0

#objective function
def obj_func(A,x,b,lmd):
    f_value = math.pow(LA.norm(np.dot(A,x.T) - b),2) / 2
    g_value = lmd * LA.norm(x,1)
    value = f_value + g_value
    return value
#--------------------------------------------------------------------------------

if __name__ == '__main__':
    n = 100
    p = 50
    s = 20   # nonzero size
    #when initialize a vector, had better to initialize it as row vector instead column vector  
    A = np.random.normal(0,1,(n,p))
    print 'A', A
    opt_x = np.array([0.0]*p)  # optimum x, used for generating b, and try to get optimum x using b and A

    #sample a list of index from 1 to 100 without duplicate
    #and create a sparse x vector with 20 non-zero elements
    random_index_list = random.sample(range(p), s) 
    for i in random_index_list:
        opt_x[i] = np.random.normal(0,10)
    print 'opt_x',opt_x
    e = np.random.normal(0,1,n)      
    print 'e',e
    b = np.dot(A,opt_x.T) + e.T
    print 'b',b


    x_k = np.array([0.0]*p)      
    lmd = math.sqrt(2*n*math.log(p))  #lambda in LOSSA objective function
    print 'lambda',lmd
    outfile = open('cgd.output','w')
    k = 50
    f=0
    success=0
    
    obj_value = obj_func(A,x_k,b,lmd)
    outfile.write(str(obj_value)+'\n')
    while(success==0):
        for i in range(p):
            A_copy = np.copy(A)
            x_copy = np.copy(x_k)
            y = soft_func((lmd/math.pow(LA.norm(A[:,i]),2)),np.dot((A[:,i]).T, b - np.dot(A_copy,x_copy.T))/np.dot((A[:,i]).T, A[:,i]))+x_copy[i]
            x_k_plus_1 = np.copy(x_k)
            x_k_plus_1[i] = y
            f_value = f_func(A,x_k_plus_1,b) 
            print 'f_value',f_value
            print i
            print 'forigin', f_func(A,x_k,b)
            if f_func(A,x_k,b) <= f_func(A,x_k_plus_1,b)+0.001:
                k=k-1
                f=1
            print'f', f
            if i==49:
                if k==0:
                    success = 1
                else:
                    k=50
            if f==0:
                x_k =  x_k_plus_1
            f=0
            print 'k', k

    a_value = np.dot(A,x_k_plus_1)
    print a_value
    print opt_x
    print e
    print x_k_plus_1
    total=0
    for k in range(p):
        total+=math.pow(b[k]-np.dot(A,x_k_plus_1)[k],2)
    total=total/p
    print total
