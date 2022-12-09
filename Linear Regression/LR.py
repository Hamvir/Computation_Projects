# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 15:36:46 2022

@author: Hamvir Dev
"""

import numpy as np
import subprocess
import re

# Function to get data from the function in a string format and convert it-
# -to numpy matrix form
def stringtodict(someVar):
    d = [] 
    temp2 = [i.group() for i in re.finditer(r"[+-]?\d+\.\d+(.*?)+\]", someVar)]
    y = np.zeros(len(temp2))
    x = []
    
    #converting data to array form
    for idx in range(len(temp2)):
        temp_y = [temp.group() for temp in re.finditer(r"[+-]?\d+\.\d+", temp2[idx])]
        temp2_y= []
        temp2_y.append(float(temp_y[0]))
        y[idx]=temp2_y[0]
        temp3 = [temp.group() for temp in re.finditer(r"\[(.*?)\]", temp2[idx])]
        temp_x= [float(s.group()) for s in  re.finditer(r"[+-]?\d+\.\d+", temp3[0])]
        x.append(temp_x)
    x = np.array(x)
    return x,y

# The below function reshapes the returned data in user usable matrices like A and Y
def data_pre_process():
    someVar = subprocess.run(["./linearregdata_windows.exe",'21337'],stdout=subprocess.PIPE).stdout.decode("utf-8") # Get string data in 'somevar' variable
    A,y = stringtodict(someVar)
    y = y.reshape((y.shape[0],1)) # 1000 is number of data points
    I = np.ones((y.shape[0],1))
    A=np.append(A,I,axis = 1)
    return(A,y)      # user usable matrix form returned

# Main function - our code to find the required answer runs here
if __name__ == '__main__':
    A,y = data_pre_process() # A matrix is the data matrix, y - output matrix 
    
    # Finding w and b:
    w=np.linalg.inv(A.T@A)@A.T@y  
    k=w.shape    #need to determine length of w
    b=w[k[0]-1][0]
    w=w[0:k[0]-1,:]
    print("The w value is:")
    print(np.round(w,decimals=3))
    print("The b value is:")
    print(np.round(b,decimals=3))