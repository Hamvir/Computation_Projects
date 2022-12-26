# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 23:05:40 2022

@author: Hamvir Dev
"""

import cvxpy as cp
import pandas as pd
import numpy as np
import cvxopt
from cvxopt import matrix
import matplotlib.pyplot as plt

# This functions solves the primal optimization problem to find
# the optimised value of w and b

def call_cvpxy(x,y,dim,n):
    w=cp.Variable(dim)
    b=cp.Variable()
    I = np.ones(n)
    d=np.zeros((n,n))
    for i in range(0,n):
        d[i,i]=y[i]
    objective = cp.Minimize(cp.sum_squares(w)/2)
    constraints = [d@(x @ w + b*I)>=I]
    prob = cp.Problem(objective,constraints)
    result = prob.solve()
    return(w.value,b.value)

# This function solves the dual optimisation problem to find 
# the optimised value of w and b and compare with solution of primal

def solve_dual(x,y,n):
     z=cp.Variable(n)
     A=np.multiply(x,y)
     P= A@A.T
     q = np.ones(n)
     h = np.zeros(n)
     
     objective = cp.Minimize((1/2)*cp.sum_squares(A.T@z) - q.T @ z)
     constraints = [z>=h,y.T@z==0]
     prob = cp.Problem(objective,constraints)
     result = prob.solve()
     x_dual=A.T@z.value
     dual_value = -(1/2)*(z.value.T@P@z.value)+q.T@z.value
     return(z.value,x_dual,dual_value)

# Plots the graph of any 2D data points and the colors refer to as follows
# square points refer to points y=-1
# circle points refer to y=+1
# Red color refers to active constraint
# Blue color refers to inactive constraint
    
def plot_svm(w,b,x,y,dim,n):
    A=np.multiply(x,y)
    I=np.ones(n)
    I=I.reshape(n,1)
    w=w.reshape(dim,1)
    active=A@w+b*y-I
    indices_active_pos=[]
    indices_active_neg=[]
    indices_pos=[]
    indices_neg=[]
    for i in range(0,n):
        if active[i,0]<=0 and y[i]==1:
            indices_active_pos.append(i)
        elif active[i,0]<=0 and y[i]==-1:
            indices_active_neg.append(i)
        elif y[i]==1:
            indices_pos.append(i)
        else :
            indices_neg.append(i)
                
            
    x_1 = np.take(x,indices_active_pos,axis=0)
    x_2 = np.take(x,indices_active_neg,axis=0)
    x_3 = np.take(x,indices_pos,axis=0)
    x_4 = np.take(x,indices_neg,axis=0)
    plt.scatter(x_1[:,0],x_1[:,1],color='r')           
    plt.scatter(x_2[:,0],x_2[:,1],color='r',marker='s')
    plt.scatter(x_3[:,0],x_3[:,1],color='b')           
    plt.scatter(x_4[:,0],x_4[:,1],color='b',marker='s')
    z=np.linspace(-3,3,2000)
    plt.plot(z,(-b-w[0]*z)/w[1])
    
if __name__ == '__main__':
    # READING DATA POINTS
    df1 = pd.read_csv("Data.csv",header = None)
    df2 = pd.read_csv("Labels.csv",header = None)
    x = df1.to_numpy()
    y = df2.to_numpy()
    dim = x.shape[1] # number of features in x
    n = x.shape[0] # number of data points
   
    # SOLVING PRIMAL AND ITS DUAL
    w_opt,b_opt = call_cvpxy(x,y,dim,n)
    lambd_a,w_dual,dual_value=solve_dual(x,y,n)
    
    # PRINTING OF VALUES
    primal_value_opt=1/2*(np.linalg.norm(w_opt)**2)
    gamma = lambd_a[0:5].T@y[0:5]
    print("The optimised points through cvpxy function:")
    print("w optimum is :",w_opt)
    print("b optimum is :",b_opt)
    print("The primal function value is :",primal_value_opt)
    print("The optimised points for dual function:")
    print("lambda is :",lambd_a)
    print("w optimum through dual function is :",w_dual)
    print("dual function value is :",dual_value)
    print("The gamma value is:",gamma)
    
    # PLOTTING GRAPH FOR 2D DATA
    if(dim==2):
        plot_svm(w_opt,b_opt,x,y,dim,n)