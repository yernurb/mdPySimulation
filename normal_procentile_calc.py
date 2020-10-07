import numpy as np 

p  = 0.3275911
a1 = 0.254829592
a2 = -0.284496736
a3 = 1.421413741
a4 = -1.453152027
a5 = 1.061405429
 
def erfc(x):
    t = 1.0/(1.0 + p*x)
    poly = a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
    return poly * np.exp(-x*x)

sigma = 15

def proc(x):
    x_val = (x - 100) / (np.sqrt(2)*sigma)
    return 0.5*erfc(x_val)
print(100*proc(145))
