import random

import numpy as np


class Particle:
    def __init__(self,
                 x_min=0,
                 x_max=1):
        self.position = np.random.uniform(x_min, x_max, PSO.dimension)
        self.velocity = np.zeros(PSO.dimension)
        self.fitness = PSO.fitness_function(self.position)
        self.best_position = self.position.copy()
        self.best_fitness = self.fitness
        self.x_min = x_min
        self.x_max = x_max

    def update_position(self,
                        w,
                        c1,
                        c2,
                        global_best_position):
        self.velocity = w * self.velocity + \
                        c1 * random.random() * (self.best_position - self.position) + \
                        c2 * random.random() * (global_best_position - self.position)
        self.position += self.velocity
        for idx, p in enumerate(self.position):
            if p > self.x_max:
                self.position[idx] = self.x_max
            elif p < self.x_min:
                self.position[idx] = self.x_min
        self.fitness = PSO.fitness_function(self.position)
        self.update_particle_best()

    def update_particle_best(self):
        if PSO.is_better(self.fitness, self.best_fitness):
            self.best_position = self.position.copy()
            self.best_fitness = self.fitness

    def __repr__(self):
        return (f'{str(self.position)} \n '
                f'{self.fitness} \n '
                f'{str(self.best_position)} \n '
                f'{self.best_fitness} \n')


class Global_Best_Particle:
    def __init__(self):
        if PSO.problem_type == 'min':
            self.fitness = float('inf')
        elif PSO.problem_type == 'max':
            self.fitness = float('-inf')
        else:
            self.fitness = None
        self.position = None

    def update_global_best(self,
                           position,
                           fitness):
        if PSO.is_better(fitness,
                         self.fitness):
            self.position = position.copy()
            self.fitness = fitness


class PSO:
    dimension = 0
    fitness_function = None
    problem_type = 'min'

    @staticmethod
    def is_better(num, ref):
        if PSO.problem_type == 'max':
            return num > ref
        elif PSO.problem_type == 'min':
            return num < ref
        else:
            return False

    def __init__(self,
                 fitness_function,
                 dimension,
                 swarm_size=10,
                 w_damp=0.9,
                 c1=2,
                 c2=2,
                 max_iter=100,
                 problem_type='min',
                 x_min=0,
                 x_max=1):
        self.w = 1
        self.w_damp = w_damp
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.x_min = x_min
        self.x_max = x_max
        self.swarm_size = swarm_size

        PSO.fitness_function = fitness_function
        PSO.dimension = dimension
        PSO.problem_type = problem_type

        self.particles = []
        self.global_best_particle = Global_Best_Particle()
        for idx in range(self.swarm_size):
            self.particles.append(Particle(self.x_min,
                                           self.x_max))
            self.global_best_particle.update_global_best(self.particles[idx].best_position,
                                                         self.particles[idx].best_fitness)

    def run(self):
        for i in range(self.max_iter):
            for particle in self.particles:
                particle.update_position(self.w,
                                         self.c1,
                                         self.c2,
                                         self.global_best_particle.position)
                self.global_best_particle.update_global_best(particle.best_position,
                                                             particle.best_fitness)

            self.w *= self.w_damp
            print(f'[info] :: {i:4} :: '
                  f'{self.global_best_particle.fitness} --> '
                  f'{self.global_best_particle.position}')

    def __repr__(self):
        strx = (f"\n{'-' * 25}\n" +
                f'PSO Config :: \n' +
                f"{'-' * 25}\n" +
                f'\t {"dimension:":15} {PSO.dimension}\n' +
                f'\t {"swarm_size:":15} {self.swarm_size}\n' +
                f'\t {"w_damp:":15} {self.w_damp}\n' +
                f'\t {"c1:":15} {self.c1}\n' +
                f'\t {"c2:":15} {self.c2}\n' +
                f'\t {"max_iter:":15} {self.max_iter}\n' +
                f'\t {"problem_type:":15} {self.problem_type}\n' +
                f'\t {"x_min:":15} {self.x_min}\n' +
                f'\t {"x_max:":15} {self.x_max}\n' +
                '-' * 25
                )
        return strx


if __name__ == "__main__":
    print('An implementation of PSO (Particle Swarm Optimization) algorithm in Python.')
