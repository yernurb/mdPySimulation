import numpy as np
import datetime

radius = 10
nx, ny, nz = 5, 5, 5

lattice = np.random.rand(nx, ny, nz)

def square(x):
    return x*x

lattice2 = map(square, lattice)

print("lattice = ", lattice)
print("lattice2 = ", list(lattice2))
