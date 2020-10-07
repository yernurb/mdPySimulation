import numpy as np
import DVector as dv


class Particle:
    def __init__(self, m=1.0, R=1.0, elastic=10.0, viscous=5.0):
        self.mass = m  # mass
        self.radius = R  # radius
        self.elastic = elastic  # elastic property
        self.viscous = viscous  # damping property
        self.rtd0 = dv.Vec3()   # position vector (zero's order time derivative of position vector)
        self.rtd1 = dv.Vec3()   # velocity vector (first order time derivative of position vector)
        self.rtd2 = dv.Vec3()   # acceleration vector (second order time derivative of position vector)
        self.rtd3 = dv.Vec3()   # third order time derivative of position vector
        self.rtd4 = dv.Vec3()   # fourth order time derivative of position vector
        self.force = dv.Vec3()  # resulting force vector acting upon the particle
    
    @property
    def m(self):
        return self.mass
    
    @property
    def R(self):
        return self.radius

    @property
    def elas(self):
        return self.elastic

    @property
    def visc(self):
        return self.viscous

    @m.setter
    def m(self, m):
        self.mass = m
    
    @R.setter
    def R(self, R):
        self.radius = R

    @elas.setter
    def elas(self, L):
        self.elastic = L
    
    @visc.setter
    def visc(self, viscous):
        self.viscous = viscous

    def kinetic_energy(self):
        return self.m*self.rtd1.value2()/2

    def __repr__(self):
        pos = self.rtd0.__repr__()
        vel = self.rtd1.__repr__()
        force = self.force.__repr__()
        return 'M: {0}, R: {1}, pos: {2}, vel: {3}, force: {4}, energy: {5}'.format(self.m, self.R, pos, vel, force,
                                                                                    self.kinetic_energy())
