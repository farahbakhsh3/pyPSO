import numpy as np
import clsPSO as PSO

if __name__ == "__main__":
    def fit(x):
        return np.sum(x ** 2)


    pso = PSO.PSO(fitness_function=fit,
              dimension=5,
              swarm_size=25,
              w_damp=0.9,
              c1=2,
              c2=2,
              max_iter=50,
              problem_type='min',
              x_min=-10,
              x_max=10)
    pso.run()
    print(pso)
    print(f'[info] :: Best solution :: '
          f'{pso.global_best_particle.fitness} --> '
          f'{pso.global_best_particle.position}')
