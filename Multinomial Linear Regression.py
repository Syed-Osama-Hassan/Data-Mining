#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
def main():
    #Reading file data
    file_x1 = pd.read_csv(r'C:\Users\Syed\Desktop\Studies\7th Semester\Data Mining\Assignments\data2.txt', usecols=[0], header= None)
    file_x2 = pd.read_csv(r'C:\Users\Syed\Desktop\Studies\7th Semester\Data Mining\Assignments\data2.txt', usecols=[1], header= None)
    file_y = pd.read_csv(r'C:\Users\Syed\Desktop\Studies\7th Semester\Data Mining\Assignments\data2.txt', usecols=[2], header= None)



    # Converting to numpy
    column1 = file_x1.to_numpy()
    column2 = file_x2.to_numpy()
    column3 = file_y.to_numpy()
    
    #Assigning x1, x2, and y
    x1 = []
    for sublist in column1:
        for item in sublist:
            x1.append(item)
    
    x2 = []
    for sublist in column2:
        for item in sublist:
            x2.append(item)
    
    
    y = []
    for sublist in column3:
        for item in sublist:
            y.append(item)
    
    #Normalizing data
    max_value = max(x1)
    mean = sum(x1) / len(x1)
    counter = 0
    for i in x1:
        new = (i - mean) / max_value
        x1[counter] = new
        counter = counter + 1
    
    max_value = max(x2)
    mean = sum(x2) / len(x2)
    counter = 0
    for i in x2:
        new = (i - mean) / max_value
        x2[counter] = new
        counter = counter + 1
    
    
    
    b0 = 0
    b1 = 0
    b2 = 0
    cost = calculate_cost(x1, x2, y, b0, b1, b2)
    print("Initial cost : ",cost)
    
    counter = 0  #Outer loop counter
    L= 0.01    # Learning rate
    m = len(x1)  # Inner loop counter
    new_cost = []
    new_cost.append(cost)
    cost_counter = 0
    
    while counter == 0:
        a, b, c = gradient_descent(x1, x2 , y, b0, b1, b2, m)
        
        # Updating theta's
        b0 = b0 - (L *(1 / m) * a)    
        b1 = b1 - (L *(1 / m) * b)
        b2 = b2 - (L *(1 / m) * c)
        
        #Calculating cost against new theta's
        cost = calculate_cost(x1, x2, y, b0, b1, b2)
        if( cost > new_cost[cost_counter]):
            print("Increasing cost")
            break
            
        
        if( new_cost[cost_counter] - cost < 1e-7):
            print("Difference between two consecutive cost is less than 1e-7, terminating loop.")
            break
        else:
            new_cost.append(cost)
            cost_counter += 1
        
        
    new_cost.append(calculate_cost(x1,x2,y,b0,b1,b2))
    print("Final cost: " ,calculate_cost(x1,x2,y,b0,b1,b2))        
    print("b0: " , b0)
    print("b1: ", b1)
    print("b2: ", b2)
    x1 = np.array(x1)
    x2 = np.array(x2)
    hypothesis = b0 + b1 * x1 + b2 * x2
    
    #Plotting file data and best fit line  
    x1_model = np.linspace(min(x1), max(x1), 30)
    x2_model = np.linspace(min(x1), max(x1), 30)
    X, Y = np.meshgrid(x1_model, x2_model)
    Z = f(X,Y,b0,b1,b2)
    
    fig = plt.figure(figsize= (12,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, y, marker='.', color='red')
    ax.contour3D(X, Y, Z, 50)
    ax.set_xlabel("X1");
    ax.set_ylabel("X2")
    ax.set_zlabel("y")
    plt.show()
    
    #Plotting iteration vs cost
    plt.plot([0, len(new_cost)], [max(new_cost), min(new_cost)])
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.show()

def f(x, y,b0,b1,b2):
    return b0 + b1 * x + b2 *y


def gradient_descent(x1, x2 , y, b0, b1, b2, m):
    a = 0
    b = 0
    c = 0

    for index in range(m):
        a += ( (b0 + b1 * x1[index] + b2 * x2[index]) - y[index]) * 1
        b += ( (b0 + b1 * x1[index] + b2 * x2[index]) - y[index]) * x1[index] 
        c += ( (b0 + b1 * x1[index] + b2 * x2[index]) - y[index]) * x2[index]

    return a,b,c

def calculate_cost(x1, x2, y, b0, b1, b2):
    counter = len(x1)
    cost = 0
    for i in range(0, counter):
        cost = cost + (y[i] - (b0 + b1 * x1[i] + b2 * x2[i]))**2
    cost = cost / (2 * counter)
    return cost
    
    
if __name__ == "__main__":
    main()


# 

# In[ ]:




