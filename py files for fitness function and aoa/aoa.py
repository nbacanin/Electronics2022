#Arithmetic optimization algorithm - AOA

import math
from utilities.solution import Solution
import random
# all random should be updated by numpy (faster calculation)
import copy
import numpy as np

class AOA:
    def __init__(self, n, function):

        # population size
        self.N = n
        self.function = function
        self.population = []
        self.best_solution = [None] * self.function.D
        self.a = 5      #alpha param
        self.mi = 0.5  #mi param
        self.moa = 0.0
        self.mop = 0.0
        self.e = 1
    
        self.dest_pos = np.zeros(self.function.D)
        self.dest_score = float("inf")
    

    def initial_population(self):
        for i in range(0, self.N):
            local_solution = Solution(self.function)
            self.population.append(local_solution)

        self.population.sort(key=lambda x: x.fitness)
        self.best_solution = copy.deepcopy(self.population[-1].x)

        # printing objective function, sorting is in the ascending order
        # for i in self.population:
        #   print(i.objective_function)

    def sort_population(self):
        self.population.sort(key=lambda x: x.fitness)
        self.best_solution = self.population[0].x

    def get_global_best(self):
        return self.population[-1].objective_function
    
    def get_global_worst(self):
        return self.population[0].objective_function

    def get_global_best_fitness(self):
        return self.population[-1].fitness
    
    def optimum(self):
        print('f(x*) = ', self.function.minimum, 'at x* = ', self.function.solution)
        
    def algorithm(self):
        return 'AOA'
    
    def objective(self):
        
        result = []
        
        for i in range(self.N):
            result.append(self.population[i].objective_function)
            
        return result
    
    def average_result(self):
        return np.mean(np.array(self.objective()))
    
    def std_result(self):        
        return np.std(np.array(self.objective()))
    
    def median_result(self):
        return np.median(np.array(self.objective()))
        
       
    def print_global_parameters(self):
            for i in range(0, len(self.best_solution)):
                 print('X: {}'.format(self.best_solution[i]))
                 
    def get_best_solutions(self):
        return np.array(self.best_solution)

    def get_solutions(self):
        
        sol = np.zeros((self.N, self.function.D))
        for i in range(len(self.population)):
            sol[i] = np.array(self.population[i].x)
        return sol


    def print_all_solutions(self):
        print("******all solutions objectives**********")
        for i in range(0,len(self.population)):

              print('solution {}'.format(i))
              print('objective:{}'.format(self.population[i].objective_function))
              print('fitness:{}'.format(self.population[i].fitness))
              print('solution {}: '.format(self.population[i].x))
              print('--------------------------------------')

# =============================================================================
# New methods
# =============================================================================

    def update_MOA(self, C_iter, Min, Max, M_Iter):
        self.moa = Min + C_iter*((Max-Min)/(M_Iter))

    def update_MOP(self, C_Iter, M_Iter):
        self.mop = 1 - (pow(C_Iter, 1/self.a))*(pow(M_Iter, 1/self.a))

    def update_position(self, C_Iter):
        for i in range(self.N):
            for j in range(self.function.D):
                X_curr = np.array(copy.deepcopy(self.population[j].x))
                r1 = np.random.random();
                r2 = np.random.random();
                r3 = np.random.random();
                if r1 > self.moa:
                    if r2 > 0.5:
                        X_curr[C_Iter + 1] = self.best_solution/(self.mop+self.e)*((self.function.ub[j]-self.function.lb[j])*self.mi+self.function.lb[j])
                    else:
                        X_curr[C_Iter + 1] = self.best_solution*self.mop*((self.function.ub[j]-self.function.lb[j])*self.mi+self.function.lb[j])
                else:
                    if r3 > 0.5:
                        X_curr[C_Iter + 1] = self.best_solution-self.mop*((self.function.ub[j]-self.function.lb[j])*self.mi+self.function.lb[j])
                    else:
                        X_curr[C_Iter + 1] = self.best_solution+self.mop*((self.function.ub[j]-self.function.lb[j])*self.mi+self.function.lb[j])

        #r1 = np.random.random()
        for i in range(self.N):

            fitness = self.population[i].fitness
            Xcurr = copy.deepcopy(self.population[i].x)

            if fitness > self.dest_score:
                self.dest_score = fitness
                self.dest_pos = Xcurr
        # =============================================================================
        # position update
        # =============================================================================
        for i in range(self.N):
            Xcurr = copy.deepcopy(self.population[i].x)
            Xnew = [None] * self.function.D
            for j in range(self.function.D):

                # Update r2, r3, and r4 for Eq. 3.3
                r2 = (2 * np.pi) * np.random.random()
                r3 = 2 * np.random.random()
                r4 = np.random.random()

                # Eq. 3.3
                if r4 < (0.5):
                    # Eq. 3.1
                    Xnew[j] = Xcurr[j] + (r1 * np.sin(r2) * np.abs(r3 * self.dest_pos[j] - Xcurr[j]))
                else:
                    # Eq. 3.2
                    Xnew[j] = Xcurr[j] + (r1 * np.cos(r2) * np.abs(r3 * self.dest_pos[j] - Xcurr[j]))

                # =============================================================================
                # check boundaries
                # =============================================================================
                if Xnew[j] < self.function.lb[j]:
                    Xnew[j] = self.function.lb[j]

                elif Xnew[j] > self.function.ub[j]:
                    Xnew[j] = self.function.ub[j]

            # =============================================================================
            # generate new solution and compare it to the old solution
            # =============================================================================

            solution = Solution(self.function, Xnew)

            if solution.fitness > self.population[i].fitness:

                self.population[i] = solution
