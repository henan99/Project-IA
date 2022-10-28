import numpy as np 
import matplotlib.pyplot as plt 

#i added a comment lol


def main():
    print("hello world")
    



def new_function(x, z):
    x = x+1
    return x**2/z

def new_func2(x):
    y = np.zeros(x)
    
    return y
y = new_func2(10)
print(y)

x = np.linspace(1,10)
plt.figure(1)
plt.plot(x,new_function(x))
plt.show()
