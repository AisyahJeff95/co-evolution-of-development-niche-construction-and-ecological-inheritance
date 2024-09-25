import os
import pickle
from evaluation import eval_fitness
import matplotlib.pyplot as plt
import numpy as np
from pureples.shared.visualize import draw_net_3d
import glob

local_dir = os.path.dirname(__file__)

data_files = [glob.glob(local_dir+f"/genomes{i}.pkl")[0] for i in range(1,11)]
genome_data = []
for file in data_files:
  with open(file, "rb") as f:
    genome_data.append(pickle.load(f))

#find total fitness
fitness=[]
for i in range (9):
    total_fitness=[]
    for genome in genome_data[i]:
        gen_fitness=[]
        for ind,g in genome:
            gen_fitness.append(g.fitness)

        gen_fitness.sort()
        total_fitness.append(gen_fitness)
    fitness.append(total_fitness)

#average
data = np.array(fitness)
average=np.average(data, axis=0) #the average of all fitness

#plot it
plt.figure()
plt.plot(np.arange(len(genome_data[0])),[np.mean(i) for i in average])
plt.legend(["Average"])
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.ylim(0,65)
plt.savefig('ave_fitness_case_.svg')
plt.show()



