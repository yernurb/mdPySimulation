import numpy as np 
from DVector import Vec3
import DVector as dv
from SphericalParticle import Particle
import copy


class World3D:
    def __init__(self, Lx=100.0, Ly=100.0, Lz=100.0, Omega=1e-4):
        self.Lx = Lx    # X - size of the world (radial direction)
        self.Ly = Ly    # Y - size of the world (azimuthal direction)
        self.Lz = Lz    # Z - size of the world (vertical direction)
        self.Omega = Omega  # orbital frequency of the Hill's box 
        self.t = 0.0    # global simulation time
        self.dt = 0.01  # integration time step
        self.Dt = 2.0*Ly / (3*Omega*Lx)     # shearing period of adjacent Hill's boxes
        self.particles = np.array([], dtype=type(Particle))   # list of particles
        self.lc = np.ndarray((1, 1, 1), dtype=int)  # lattice structure
        self.lc_neighbours = np.ndarray((1, 1), dtype=type((int, int, int)))  # neighbour cells of all particles
        self.nx = 0     # lattice size in X direction
        self.ny = 0     # lattice size in Y direction
        self.nz = 0     # lattice size in Z direction
        self.delta = 0  # lattice cell size
        self.nb_radius = 0  # close neighbourhood radius (in lattice cell numbers)
        self.r_min = 0  # radius of the smallest particle
        self.r_max = 0  # radius of the largest particle
        self.diag = np.sqrt(Lx**2 + Ly**2 + Lz**2)  # diagonal size of the world (maximum distance between particles)

    # populate the world with size-uniform N particles with given initial kinetic energy
    def populate(self, N, m, r, elas, visc, energy):
        # for uniform size distribution, r_min = r_max = r
        self.r_min = self.r_max = r

        # estimate the initial distance between two adjacent particles, in order to
        # exclude initial overlapping
        delta = 2.5*r

        # first particle's coordinate is (delta, delta, Lz/2)
        # In the vertical Z direction we the ring plane to be located at z = Lz/2
        x, y, z = delta, delta, self.Lz / 2

        # initial thermal speed of all particles
        v = np.sqrt(2*energy/m)

        for i in range(N):
            p = Particle(m, r, elas, visc)
            pos = Vec3(x, y, z)

            # the initial velocity vector has a constant magnitude given by thermal speed
            # but random spacial orientation
            theta = np.pi*np.random.random()
            phi = 2*np.pi*np.random.random()
            vx = v*np.sin(theta)*np.cos(phi)
            vy = v*np.sin(theta)*np.sin(phi)
            vz = v*np.cos(theta)
            vel = Vec3(vx, vy, vz)
            p.rtd0, p.rtd1 = pos, vel
            self.particles = np.append(self.particles, p)

            # estimate the initial position of the next particle, adjacent
            # to the current one. If the XY plane is completely filled
            # the insertion process is stopped
            x += delta
            if x > self.Lx - delta:
                x = delta
                y += delta
                if y > self.Ly - delta:
                    print(f"Too many particles. Only N={i} particles are inserted.")
                    break

        self.create_lattice()

    # return the cell index of given spacial position (in Vec3)
    def cell_index(self, pos):
        return int(np.floor(pos.x/self.delta)), int(np.floor(pos.y/self.delta)), int(np.floor(pos.z/self.delta))

    # fill lattice according to the current position of particles
    def update_lattice(self):
        self.lc.fill(-1)
        for i, p in np.ndenumerate(self.particles):
            idx = self.cell_index(p.rtd0)
            pid, = i
            self.lc[idx] = pid

    # initialize the lattice structure after populating the world
    def create_lattice(self):
        # cell size estimated from the size of smallest particle
        self.delta = np.sqrt(2)*self.r_min

        # estimate the radius of neighbour search
        self.nb_radius = int(2*np.ceil(self.r_max / self.delta))

        # lattice size in X, Y, Z direction is calculated
        self.nx = int(np.ceil(self.Lx / self.delta))
        self.ny = int(np.ceil(self.Ly / self.delta))
        self.nz = int(np.ceil(self.Lz / self.delta))

        # resize the world size in order to exactly match cell size times lattice size
        self.Lx = self.nx * self.delta
        self.Ly = self.ny * self.delta
        self.Lz = self.nz * self.delta
        self.diag = np.sqrt(self.Lx**2 + self.Ly**2 + self.Lz**2)

        # resize the lattice, neighbour list and initialize according to the initial position of particles
        self.lc = np.resize(self.lc, (self.nx, self.ny, self.nz))
        nb_list_size = (2*self.nb_radius+1)**3-1
        self.lc_neighbours = np.resize(self.lc_neighbours, (self.particles.size, nb_list_size))
        self.update_lattice()
        for i, _ in np.ndenumerate(self.particles):
            pid, = i
            self.find_neighbours(pid)

    # fills lattice neighbour list for the given particle pid
    def find_neighbours(self, pid):
        idx = 0
        for n in range(-1, 2):
            for m in range(-1, 2):
                pos, _ = self.get_particle_image(pid, n, m)
                x, y, z = self.cell_index(pos)
                for di in range(-self.nb_radius, self.nb_radius+1):
                    for dj in range(-self.nb_radius, self.nb_radius+1):
                        for dk in range(-self.nb_radius, self.nb_radius+1):
                            if di == dj == dk == 0:
                                continue
                            i, j, k = x + di, y + dj, z + dk
                            if self.is_valid_lattice_index(i, j, k):
                                self.lc_neighbours[pid, idx] = (i, j, k)
                                idx += 1

    # returns an image (n,m) of a particle
    def get_particle_image(self, pid, n, m):
        p = self.particles[pid]
        p_image = copy.deepcopy(p)
        tau = np.mod(self.t, self.Dt)
        p_image.rtd0.x = p.rtd0.x + n*self.Lx
        p_image.rtd0.y = p.rtd0.y + m*self.Ly - (3/2)*n*self.Lx*self.Omega*tau

        p_image.rtd1.y = p.rtd1.y - (3/2)*m*self.Omega*self.Lx

        return p_image

    # checks whether given index (i,j,k) belongs to the lattice
    def is_valid_lattice_index(self, i, j, k):
        return 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz
            
    # returns indices of the closest image of a particle to the target particle
    def get_closest_image_of_particle(self, pid_target, pid):
        min_distance = self.diag
        pos_target = self.particles[pid_target].rtd0
        idx_n, idx_m = -2, -2
        for n in range(-1, 2):
            for m in range(-1, 2):
                pim = self.get_particle_image(pid, n, m)
                distance = dv.dif(pos_target, pim.rtd0).value()
                if distance < min_distance:
                    min_distance = distance
                    idx_n, idx_m = n, m
        return idx_n, idx_m

    # calculates force between two given particles
    def calculate_force_between_particles(self, p1, p2):
        n = dv.dif(p1.rtd0, p2.rtd0)
        dist = n.value()
        xi = p1.radius + p2.radius - dist
        if xi > 0:
            gamma = (p1.viscous + p2.viscous)/2
            kappa = (p1.elastic + p2.elastic)/2
            n.normalize()
            g = dv.dif(p1.rtd1, p2.rtd1)
            xidot = dv.dot(g, n)
            force = -gamma*xidot - kappa*xi
            return dv.mul(n, force)
        n.nullify()
        return n

    # prepares images for possible collision partners and calls calculate_force_between_particles to compute force
    def calculate_force(self, pid_1, pid_2):
        n1, m1 = self.get_closest_image_of_particle(pid_1, pid_2)
        n2, m2 = self.get_closest_image_of_particle(pid_2, pid_1)
        p1 = self.particles[pid_1]
        p2 = self.particles[pid_2]
        p1_partner = self.get_particle_image(pid_2, n1, m1)
        p2_partner = self.get_particle_image(pid_1, n2, m2)
        f1 = self.calculate_force_between_particles(p1, p1_partner)
        f2 = self.calculate_force_between_particles(p2, p2_partner)
        p1.force.add(f1)
        p2.force.add(f2)

    # predictor
    def predict(self):
        pass

    # TODO corrector
    # TODO make a single step
    # TODO system state analysis functions
