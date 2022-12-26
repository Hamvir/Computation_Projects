---
mathjax: true
---
# Support Vector Machine

## Problem Statement
We are given a set of data points and their classes(binary) we have to create a SVM to classify future points.

## Input format:
Give data in the format of a csv file each for x(inputs) and y(labels). 2 example files have been uploaded.

## Solution
Support vector machines (SVMs) are among the most widely used techniques for classifying data, and are very well studied. The SVM is a linear classifier, that 
is, we aim to find a function $$y = f(x) = sign(w^T x + b)$$ 
WHich classifies the datapoint to 2 classes (+1) and (-1).
The problem can be converted to following primal problem:
$$w^{opt},b^{opt}= arg min \ 1/2 \  ||w_2^2||$$
$$s.t. \ \ y_i(w^T x_i + b) >=1 \ \ \ \ \ \ for \ i = 1,....,N$$
We have solved this primal problem in the code to find the optimum w and b.
We can also solve its dual problem to find optimum w and b, the dual problem is:
$$g(\Lambda) \ = \ \Lambda^Tb \ + \ 1/2 \ \Lambda^TA\Lambda$$
$$\Lambda \ = \(\lambda_1,....\lambda_k)$$

We can also show:
$$\sum_{i:y_i=1}\lambda_i \ = \ \sum_{i:y_i=-1}\lambda_i \ = \ \gamma$$

We can also check dual function value = primal function value thus satisfying strong duality.

## How to use:
>Download the py file.

>open it any python compiler like spyder

>The user should have 2 csv files namely input data points and labels (or outputs) of those inputs.

>Read the output

>The output contains informations like optimal w and b, lambda, dual and primal function value.

>If your data points have 2 features then a graph will also be plotted for the classifier showing the separation. 
