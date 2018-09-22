import numpy as np
import math

# this is the polynomial taking coefficients
# and input and outputting the evaluated function 
def f(coeffs, x):
    y = 0
    for i in range(len(coeffs)):
        y += coeffs[i]*x**i
    return y


# evaluates fitness (RMSE) of individual
def eval_fit_ind(ind,target,points):
    error = 0
    for i in range(len(points)):
        error += (f(ind, points[i]) - f(target,points[i]))**2
        
    return math.sqrt(error)

# evaluates fitness of population
def eval_fit_pop(pop,target,points):
    fit_vals = []
    for i in range(len(pop)):
        fit_vals.append(eval_fit_ind(pop[i],target,points))
        
    return np.array(fit_vals)

# ranks population
def rank_pop(pop,target,points):
    ranked =  [ pop[i] for i in np.argsort(eval_fit_pop(pop,target,points))]
    return ranked

# crossovers
def cross_pop(pop):
    new_pop = []
    #some children are averages 
    for i in range(int(len(pop)/4)):
        for j in range(i+1,int(len(pop)/4)):
            if(len(new_pop)<len(pop)/2):
                new_pop.append((np.array((pop[i]+pop[j]))/2).astype(int))
            else: 
                break
            
    #some children alternate adopting values from each parent
    for k in range(len(pop)-len(new_pop)):
        new_pop.append(np.array([(l%2)*pop[k][l]+((l+1)%2)*pop[k+1][l] for l in range(len(pop[0]))]))
    return new_pop

# mutations
def mut_pop(pop,k):       # 1/k is (approx) prob of mutating an individual
    for i in range(len(pop)):
        x = np.random.randint(0,k)
        if(x==0):
            y = np.random.randint(0,len(pop[0]))
            z = np.random.randint(-2,3)
            # impose restraints on coeffs (search space)
            if(abs(pop[i][y]+z) <11):
                pop[i][y] = pop[i][y]+ z
    return pop
                   
        

