# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 16:11:53 2022

@author: Hamvir Dev
"""
import numpy as np
import subprocess
import re
import matplotlib as plt

# The below function gets data from required function and converts it to -
# -user usable matrices and returns it
def data_pre_process():
    someVar = subprocess.run(["./pcadata_windows.exe", '21337'],stdout=subprocess.PIPE).stdout.decode("utf-8") # reading data from function
    d = []
    x=someVar
    x = x.replace("[", "").replace("]", "").strip()
    x = re.split("\\s{1,}", x)
    for i in range(len(x)):
        temp = []
        temp = [float(idx) for idx in x[i].split(',')]
        d.append(temp)
    d= np.array(d)
    return d      # returning user usable data format

# Main function - this contains the code to run PCA
if __name__ == '__main__':
    X = data_pre_process();
    data_size = X.shape[0]
    k=int(input("Input final number of columns you want")) #Reduced column number you want
    # PCA:
    num = X - np.mean(X, axis=0) # Mean standardization
    den = np.std(X, axis=0)
    X_st = num / den             # Variance standardization
    covariance_mat = 1/data_size * (X_st.T@X_st)  # Co variance matrix
    print("1) The covariance matrix is:")
    print(np.round(covariance_mat,decimals=3))
    
    #Eigen values and eigen vectors:
    eigen_values,eigen_vectors = np.linalg.eig(covariance_mat)
    eig_val = eigen_values[0:k]
    eig_vec = eigen_vectors[0:k,:]
    print("2) Two top eigen vectors are:")
    print(np.round(eig_vec,decimals=3))
    print("3) Two top eigen values corresponding to these vectors are:")
    print(np.round(eig_val,decimals=3))
    
    
    X_proj = np.dot(X,eig_vec.T)  # Finding projection of data points on eigen vectors
    print("4) The transformed data is:")
    print(np.round(X_proj,decimals=3))
    
    
    
    
    
    
    