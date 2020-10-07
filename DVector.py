import numpy as np 


class Vec3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.vec = np.array([x, y, z])

    @property
    def x(self):
        return self.vec[0]
    
    @property
    def y(self):
        return self.vec[1]
    
    @property
    def z(self):
        return self.vec[2]

    @x.setter
    def x(self, x):
        self.vec[0] = x

    @y.setter
    def y(self, y):
        self.vec[1] = y
    
    @z.setter
    def z(self, z):
        self.vec[2] = z

    def __repr__(self):        
        return '({0}, {1}, {2})'.format(self.x, self.y, self.z)

    def nullify(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

    def value(self):
        return np.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)

    def value2(self):
        return self.x*self.x + self.y*self.y + self.z*self.z

    def add(self, v):
        self.x += v.x
        self.y += v.y
        self.z += v.z
    
    def sub(self, v):
        self.x -= v.x
        self.y -= v.y
        self.z -= v.z
    
    def scale(self, n):
        self.vec *= n

    def normalize(self):
        self.vec /= self.value()


def sum(v1, v2):
    arr = v1.vec + v2.vec
    return Vec3(arr[0], arr[1], arr[2])


def dif(v1, v2):
    arr = v1.vec - v2.vec
    return Vec3(arr[0], arr[1], arr[2])


def mul(v, n):
    arr = v.vec*n
    return Vec3(arr[0], arr[1], arr[2])


def unit(v):
    arr = v.vec/v.value()
    return Vec3(arr[0], arr[1], arr[2])


def dot(v1, v2):
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z


def cross(v1, v2):
    arr = np.array([v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x])
    return Vec3(arr[0], arr[1], arr[2])
