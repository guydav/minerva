# Code for exploring the Central Limit Theorem
# Need NumPy version 1.7 or greater and Matplotlib version

# Uncomment the next line if using IPython for inline plotting
# %matplotlib inline

import numpy as np
import random
import matplotlib.pyplot as plt

def create_pop(n):
    # A
    distrib = 1000 * np.random.random_sample((n, ))
    return distrib

def graph_distribution(dist, name):
    plt.hist(dist)
    plt.title(name)
    plt.show()

def sample_procedure(dist):
    sample_size = int(raw_input("Enter the sample size:"))
    sample_repeats = int(raw_input("Enter the number of times to repeat the experiment:"))
    sample_means = []
    for _ in range(sample_repeats):        
        # B
        sample = np.random.choice(dist, (sample_size, ))
        this_mean = np.mean(sample)
        sample_means.append(this_mean)
    
    print "Here's what the sample mean distribution looks like."
    
    plt.hist(sample_means)
    plt.title('Sample Means')
    plt.show()
    
    print "The mean of the sample means is:", np.mean(sample_means)
    print "The standard deviation of the sample means is:", np.std(sample_means)
    print ""
    
#####################################

print "First create a (pseudo)-random distribution."
population_size = int(raw_input("Enter a population size:"))
print "..."
population = create_pop(population_size)

print "Here's what the population distribution looks like."
graph_distribution(population, 'Population Distribution')

print "The population mean is:", np.mean(population) 

print "For the second step, enter the size of the samples to draw --with replacement-- from the population distribution, and how many times to repeat this procedure in order to create a distribution of sample means."

sample_flag = True

while sample_flag:
    sample_procedure(population)
    print "Perform sampling procedure again? Note that if no, then the population distribution will be lost."
    decision = raw_input("Type y or n:")
    if decision == 'n':
        sample_flag = False
    else:
        print "Doing procedure again."
        

    



