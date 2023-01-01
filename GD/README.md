---
mathjax: true
---
# Fixed step-size Gradient Descent Algorithm

## About
Here optimum point(minima) of any Lipschitz smooth function whose lipschitz constant is given is found. A plot showing how function value and gradient value changes
as the algo proceeds is also plotted for first 100 iterations(One can change the number in code)

## Input format:
Here the user is asked to give the following inputs:
1) x - starting point where to start the grad_descent algorithm.
2) L - Lipschitz constant of function.
3) epsilon - It is the value of gradient upon reaching which we want to stop.
4) max_itr - Maximum iteration upto which we want to run the algo.

Besides this include input functions(sample functions given) that give function value and gradient value respectively of a point(when a point is passed).
The output values should have the following format:

For function value it should return a string value like : "22.55"

For gradient value it should return in following format : "[2.5,3.5,......1,2]"

## How to use:
>Download the py file.

>open it any python compiler like spyder

>The user should have 2 exe files that return function value and gradient value at a given point(sample functions are given). Make sure to change the name of your file 
accordingly or replace the name in the code(see code to see where to replace)

>Read the output

>The output contains informations like optimalpoint, function value etc.
