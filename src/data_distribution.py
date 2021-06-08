'''
observe the logp scores' distribution
'''

import matplotlib.pyplot as plt 
file = "data/zinc_LogP.txt"


with open(file, 'r') as fin:
	lines = fin.readlines() 

score_lst = [float(line.strip().split()[1]) for line in lines]
score_lst = list(filter(lambda x:-10<x, score_lst))
plt.hist(score_lst, bins = 50)
plt.savefig("figure/logp.png")



