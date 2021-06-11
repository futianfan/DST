'''
start from 'C'


'''

### 1. import
import os
import numpy as np 
from time import time
from tqdm import tqdm 
from matplotlib import pyplot as plt
import pickle 
from random import shuffle 
import torch
import torch.nn as nn
import torch.nn.functional as F
from tdc import Oracle
torch.manual_seed(1)
np.random.seed(2)
from tdc import Evaluator


jnk = Oracle(name = 'JNK3')
gsk = Oracle(name = 'GSK3B')
def oracle(smiles):
	scores = jnk(smiles), gsk(smiles)
	return np.mean(scores)





diversity = Evaluator(name = 'Diversity')
kl_divergence = Evaluator(name = 'KL_Divergence')
fcd_distance = Evaluator(name = 'FCD_Distance')
novelty = Evaluator(name = 'Novelty')
validity = Evaluator(name = 'Validity')
uniqueness = Evaluator(name = 'Uniqueness')


file = "data/clean_zinc.txt"
with open(file, 'r') as fin:
	lines = fin.readlines() 
zinc_lst = [line.strip() for line in lines]
from tdc import Oracle
oracle_lst = ['JNK3', 'GSK3B']

## 5. run 
if __name__ == "__main__":

	# result_file = "result/denovo_from_" + start_smiles_lst[0] + "_generation_" + str(generations) + "_population_" + str(population_size) + ".pkl"
	# result_pkl = "result/ablation_dmg_topo_dmg_substr.pkl"
	# pkl_file = "result/denovo_qedlogpjnkgsk_start_ncncccn.pkl"
	pkl_file = "result/denovo_jnkgsk.pkl"
	idx_2_smiles2f, trace_dict = pickle.load(open(pkl_file, 'rb'))

	topk = 10
	whole_smiles2f = dict()
	for idx, (smiles2f,current_set) in tqdm(idx_2_smiles2f.items()):
		whole_smiles2f.update(smiles2f)

	smiles_f_lst = [(smiles,f) for smiles,f in whole_smiles2f.items()]
	smiles_f_lst.sort(key=lambda x:x[1], reverse=True)
	best_smiles_lst = [smiles for smiles,f in smiles_f_lst[:topk]]
	best_f_lst = [f for smiles,f in smiles_f_lst[:topk]]
	avg, std = np.mean(best_f_lst), np.std(best_f_lst)
	print('f scores', str(avg)[:5], str(std)[:5])
	#### evaluate novelty

	# t1 = time()
	# nov = novelty(best_smiles_lst, zinc_lst)
	# t2 = time()
	# print("novelty", nov, "takes", str(int(t2-t1)), 'seconds')

	#### evaluate diversity 
	t1 = time()
	div = diversity(best_smiles_lst)
	t2 = time()
	print("diversity", div, 'takes', str(t2-t1), 'seconds')


	### evaluate mean of property 
	for oracle_name in oracle_lst:
		oracle = Oracle(name = oracle_name)
		scores = oracle(best_smiles_lst)
		avg = np.mean(scores)
		std = np.std(scores)
		print(oracle_name, str(avg)[:7], str(std)[:7])




	for smiles in best_smiles_lst:
		print(smiles, str(jnk(smiles))[:5], str(gsk(smiles))[:5])

'''

Cc1ncn(-c2cccc(-c3ccnc(Nc4ccc(-n5ccc(-c6ccc([SH]=C7C=CN=N7)cn6)n5)cc4)n3)c2)n1 0.8999,0.95,0.85
Cc1ncn(-c2cccc(-c3ccnc(Nc4ccc(-n5ccc(-c6ccc(S)cn6)n5)cc4)n3)c2)n1 0.895,0.94,0.85
Cc1ncn(-c2cccc(-c3ccnc(Nc4ccc(-n5ccc(-c6ccccn6)n5)cc4)n3)c2)n1 0.865,0.94,0.79
Cc1ncn(-c2cccc(-c3ccnc(Nc4ccc(-n5cccn5)cc4)n3)c2)n1 0.8799,0.94,0.82
c1cc(-c2ccnc(Nc3ccc(-n4cccn4)cc3)n2)cc(-n2cncn2)c1 0.8200,0.87,0.77
c1cnn(-c2ccc(Nc3nccc(-c4ccc(-n5cnnc5)cc4)n3)cc2)c1 0.87,0.88,0.86
c1ccc(-c2ccnc(Nc3ccc(-n4cccn4)cc3)n2)cc1 0.865,0.87,0.86
C1=CC(c2ccnc(Nc3ccc(-n4cccn4)cc3)n2)CN1 0.6950,0.61,0.78
C1=CC(c2ccnc(Nc3ccccc3)n2)CN1 0.615,0.44,0.79
C1=CC(c2ccnc(Nc3cnccn3)n2)CN1 0.515,0.36,0.67
Nc1nccc(C2C=CNC2)n1 0.125,0.02,0.23

'''


