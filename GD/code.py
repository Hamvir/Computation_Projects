# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 08:53:18 2023

@author: Hamvir Dev
"""
import numpy as np
import subprocess
import re
import random
import matplotlib.pyplot as plt

# This function returns function value at any given point, If you want to add 
# your function just replace your function name  with "generateFval.exe"
# in the given code
def calc_func_value(x_k):
    input = "[" + ','.join(str(elem) for elem in x_k) + "]"
    someVar = subprocess.run(["./generateFval.exe", input],stdout=subprocess.PIPE).stdout.decode("utf-8")
    x_0=float(someVar)
    return(x_0)

# This function returns gradient of a function value at any given point
# If you want to add your function just replace your function name  with "generateFval.exe"
# "generateGradient.exe" in the given code
def calc_grad(x_k):
    input = "[" + ','.join(str(elem) for elem in x_k) + "]"
    someVar = subprocess.run(["./generateGradient.exe", input],stdout=subprocess.PIPE).stdout.decode("utf-8")
    someVar = re.sub(r'\r\n','',someVar)
    someVar = someVar.lstrip('[')
    someVar = someVar.rstrip(']')
    st = someVar.split(",")
    x_0=[float(i) for i in st]
    x_0=np.array(x_0)
    return(x_0)

# This function calculates the optimum point using fixed step gradient descent 
# This takes an input array, Lipschitz constant, epsilon value and max_iteration
# value as input and gives out the optimal point as output
def fixed_step_grad_descent(x_k,L,epsilon,max_itr):
    alpha = 1/L # fixed step size
    func_x=[]
    norm_x=[]
    x_v=[]
    for i in range(max_itr):
        grad_fx = calc_grad(x_k)
        fx = calc_func_value(x_k)
        if np.linalg.norm(grad_fx)<=epsilon :
            break
        x_k = x_k - alpha * grad_fx  #Updating next point
        if(i<101):
            func_x.append(fx)
            norm_x.append(np.linalg.norm(grad_fx))
            x_v.append(i)

    print("No of iterations to reach ",epsilon," norm grad fx is:")
    print(i)
    print("The optimum value of point is:")
    print(x_k)
    print("The optimum function value is:")
    print(fx)
    print("Gradient at this point is")
    print(np.linalg.norm(grad_fx))
    return(func_x,norm_x,x_v)

if __name__ == '__main__':
    x=input("Enter starting point")#Example input - 1 2 3 4 5
    x=x.split(" ")
    x=list(map(int,x))
    x=np.array(x)
    
    L=int(input("Enter L"))
    epsilon=float(input("Enter epsilon"))
    max_itr=int(input("Enter max_iteration"))
    f1,grad_f1,x_v=fixed_step_grad_descent(x,L,epsilon,max_itr)
    
    # Plotting function value and gradient value for 1st 100 iterations

    plt.plot(x_v,f1,label = 'function value vs x_k')
    plt.xlabel('no of iterations')
    plt.ylabel('fx')
    plt.legend()
    plt.show()
    plt.plot(x_v,grad_f1,label = 'gradient value vs x_k')
    plt.xlabel('no of iterations')
    plt.ylabel('grad_fx')
    plt.legend()
    plt.show()