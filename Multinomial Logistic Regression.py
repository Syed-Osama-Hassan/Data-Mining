#!/usr/bin/env python
# coding: utf-8

# In[47]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    # Reading data
    data = pd.read_csv(r"C:\Users\Syed\Desktop\Studies\7th Semester\Data Mining\Assignments\data.csv", header= None)
    # Reading output variable
    Y = pd.read_csv(r"C:\Users\Syed\Desktop\Studies\7th Semester\Data Mining\Assignments\data.csv", usecols=[400], header= None)
    
    # Converting to numpy from data frame
    data_matrix = data.to_numpy()
    Y = Y.to_numpy()
    
    # Creating matrix of features by deleting output variable column
    X = data_matrix
    X = np.delete(X, 400, axis = 1)
    #Inserting 1 to the 1st column of X for x0
    X = np.insert(X,0,1, axis = 1)
    
    # Creating 10 data sets from output variable 
    y_test = np.zeros((Y.shape[0], 10))
    for i in range(10):
        y_test[:, i] = np.where(Y[:,0] == i, 1, 0)
    
    
    m = len(X)    # No. of training data
    L = 0.0001    # Learning rate
    
    # Initializing theta with zeroes ( 10 * 401) matrix
    theta = np.zeros((10,X.shape[1]))
    
    # Loop to train each data set
    for i in range(10):
        count = 0
        cost = []
        cost_counter = 0
        cost.append(calculate_cost(X, y_test[:, i], theta[i,:], m))
        
        print("Training data set: ", i)
        print("Initial cost: ", cost[cost_counter])
        
        # Loop to find the gradient that will reduce the cost
        while count == 0:
            theta[i,:] = gradient_descent(X, y_test[:,i], L, theta[i,:], m)
    
            res = calculate_cost(X, y_test[:, i],theta[i,:], m)
        
            if(res > cost[cost_counter]):
                print("Increase in cost")
                break
            
            if(cost[cost_counter] - res < 1e-4):
                break
            else:
                cost_counter += 1
                cost.append(res)
      
        print("Final cost: ", cost[cost_counter])
        print("Accuracy: ", accuracy(X, y_test[:, i], theta[i, :]))    
        print("")
        
        #Plotting iterations vs cost
        plt.plot([0, len(cost)], [max(cost), min(cost)])
        plt.xlabel('iteration')
        plt.ylabel('cost')
        plt.show()
    
    # Printing One-vs-All accuracy
    a = accuracy_OneVsAll(X, theta, Y)
    print("One-Vs-All Accuracy: " , a)

# Function that will return the overall accuracy of the models
def accuracy_OneVsAll(x, theta, y):
    m = x.shape[0]
    count = 0
    for i in range(m):
        res = []
        xi = x[i] 
        for j in range(10):
            
            val = h(xi, theta[j,:])
            res.append(val)
        
        if(res.index(max(res)) == y[i]):
            count += 1
    
    return (count / m) * 100
     
# Function will return the individual accuracy of each model      
def accuracy(x, y, theta):
    cost= 0
    correct = 0
   
    for i in range(len(x)):
        pc = 2
        xi = x[i]
        cost = h(xi, theta)
        if(cost >= 0.5):
            pc = 1.0
            
        else:
            pc = 0.0
        if(pc == y[i]):
            correct += 1
        
    return (correct / len(y)) * 100
    
# Function that will return the new theta's
def gradient_descent(x, y, L, theta, m):
    
    x = x.T
    theta = theta.T
    dot_product = np.dot(theta, x)
    h = signoid(dot_product)
    diff = h - y
    diff = diff.T
    pDerivative = np.dot(x, diff)
    
    newTheta = theta - L * pDerivative 
    
    return newTheta

# Signoid function
def signoid(z):
    return 1 / (1 + np.exp(-z))

# Function that calculate dot product and pass it to signoid 
def h(x, theta):
    theta = theta.T
    
    return signoid((theta @ x) )

# Cost function
def calculate_cost(x,y, theta, m):
    cost = 0
    
    for i in range(m):
        xi = x[i]
        yi = y[i]
        cost += -(np.sum((yi * (np.log(h(xi, theta)))) + ((1 - yi) * (np.log(1 - (h(xi, theta)))))))        
    return cost / m

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




