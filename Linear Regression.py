#!/usr/bin/env python
# coding: utf-8

# In[60]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    #Reading file data
    file = pd.read_csv(r'C:\Users\Syed\Desktop\Studies\7th Semester\Data Mining\Assignments\data1.txt', usecols=[0], header= None)
    file_y = pd.read_csv(r'C:\Users\Syed\Desktop\Studies\7th Semester\Data Mining\Assignments\data1.txt', usecols=[1], header= None)



    # Converting to numpy
    column1 = file.to_numpy()
    column2 = file_y.to_numpy()
    
    #Assigning x and y
    x = []
    for sublist in column1:
        for item in sublist:
            x.append(item)
            
    y = []
    for sublist in column2:
        for item in sublist:
            y.append(item)
            
    #Normalizing data
    max_value = max(x)
    mean = sum(x) / len(x)
    counter = 0
    for i in x:
        new = (i - mean) / max_value
        x[counter] = new
        counter = counter + 1
    
    new_b0 = 0
    new_b1 = 0
    m = len(x)
    cost = calculate_cost(x, y, new_b0, new_b1,m)
    print("Initial cost : ",cost)
    
    counter = 0
    L= 0.001
    cost_counter = 0
    new_cost = []
    new_cost.append(cost)
    cond = 1e-7
   
    while counter == 0:
        a, b = gradient_descent(x, y, new_b0, new_b1, m)
        
        
        new_b0 -= (L *(1 / m) * a )   
        new_b1 -= (L * (1 / m) * b )
        
        res = calculate_cost(x,y,new_b0,new_b1,m)
        
        if(res > new_cost[cost_counter]):
            print("Incresing in cost.")
            break
        
        if(new_cost[cost_counter] - res <= cond):
            print("Difference between two consecutive cost is less than 1e-7, terminating loop.")
            break
        else:
            cost_counter += 1
            new_cost.append(res)
        
        
        
        
        
       
    x = np.array(x)
    y = np.array(y)      
    print("Final Cost: ", new_cost[cost_counter] )    
    print("b0: " , new_b0)
    print("b1: ", new_b1)
    
    #Plotting file data and best fit line
    
    hypothesis = new_b0 + (new_b1 * x)
    plt.plot(x, hypothesis, color='red')
    plt.scatter(x,y, marker='o')
    plt.title('Linear Regression')
    plt.xlabel('population')
    plt.ylabel('profit')
    plt.show()
    
    
    
    #Plotting iteration vs cost
    plt.plot([0, len(new_cost)], [max(new_cost), min(new_cost)])
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.show()

def gradient_descent(x , y, b0, b1, m):
    a = 0 
    b = 0
    for index in range(0, m):
        a += ( b0 + b1 * x[index] - y[index]) * 1 
        b += ( b0 + b1 * x[index] - y[index]) * x[index]
    return a, b
    
def calculate_cost(x,y, b0, b1, m):
    cost = 0
    for i in range(0,m):
        cost += (b0 + b1 * x[i] - y[i])**2
    cost = cost / (2 * m)
    return cost
    
    
if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




