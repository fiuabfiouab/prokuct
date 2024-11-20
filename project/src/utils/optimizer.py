import numpy as np

class PSOOptimizer:
    def __init__(self, n_particles, n_dimensions):
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.particles = np.random.rand(n_particles, n_dimensions)
        self.velocities = np.zeros((n_particles, n_dimensions))
        self.best_positions = self.particles.copy()
        self.global_best = self.particles[0]
        
    def optimize(self, fitness_func, max_iterations=100):
        for _ in range(max_iterations):
            # Update particle positions and velocities
            self.update_particles(fitness_func)
        return self.global_best
    
    def update_particles(self, fitness_func):
        # Update particle velocities and positions
        self.velocities = 0.5 * self.velocities + \
                         0.2 * np.random.rand(*self.particles.shape) * \
                         (self.best_positions - self.particles) + \
                         0.2 * np.random.rand(*self.particles.shape) * \
                         (self.global_best - self.particles)
        self.particles += self.velocities