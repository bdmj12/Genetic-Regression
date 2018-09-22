import numpy as np
import matplotlib.pyplot as plt

from Functions import *

pop_size = 50
degree = 13

if __name__ == '__main__':
    
    #create target coefficients
    target = np.array([])
    for i in range(degree+1):
        target = np.append(target, np.random.randint(low=-10,high=11)).astype(int)

    #generate points
    points = np.linspace(-20,20,num=100)

    # create a (random) population
    pop = []
    for i in range(pop_size):    
        pop.append(np.random.randint(low=-10,high=11, size=(degree+1)))
    

    # run the algorithm and finds an optimum
    
    g = 0 #generations
    mut_prob = 3   # probability of a mutation in a given individual (i.e. 1/mut_prob)
    best_fitn = np.amin(eval_fit_pop(pop,target,points)) # 0 is optimal!
    
    while(best_fitn > 1 and g<1000):          
            
        pop = rank_pop(pop,target,points)
        pop = cross_pop(pop)
        pop = mut_pop(pop,mut_prob)
        
        best_fitn = np.amin(eval_fit_pop(pop,target,points))
        
        print("Generation: " + str(g))
        print("Least error: " + str(int(best_fitn)))
        print("Coeff difference: ")
        print(target-pop[0])  # best would be all zeros
        
        if(g%100==0):
                plt.plot(points, [f(target,x) for x in points],'r+')
                plt.plot(points, [f(pop[0],x) for x in points],'bo')
                plt.xlim(-20,20)
                plt.ylim(min([f(target,x) for x in points]),max([f(target,x) for x in points]))
                plt.show()
            
        g=g+1
            
    print("\n")
    print("Completed at generation: " + str(g))
    print("Least error: " + str(best_fitn))
    pop = rank_pop(pop,target,points)
    
    print("Target: ")
    print(target)
    print("Best individual is: ")
    print(pop[0])
            
    plt.plot(points, [f(target,x) for x in points],'r+')
    plt.plot(points, [f(pop[0],x) for x in points],'bo')
    plt.xlim(-20,20)
    plt.ylim(min([f(target,x) for x in points]),max([f(target,x) for x in points]))
    plt.show()
            
            

            