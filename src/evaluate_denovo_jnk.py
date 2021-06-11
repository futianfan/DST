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

from chemutils import * 
## 2. data and oracle 
qed = Oracle(name = 'qed')
logp = Oracle(name = 'logp')
jnk = Oracle(name = 'JNK3')
gsk = Oracle(name = 'GSK3B')
def foracle(smiles):
	return logp(smiles)

prop = 'jnk'


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
oracle_lst = ['jnk3']

## 5. run 
if __name__ == "__main__":

	# result_file = "result/denovo_from_" + start_smiles_lst[0] + "_generation_" + str(generations) + "_population_" + str(population_size) + ".pkl"
	# result_pkl = "result/ablation_dmg_topo_dmg_substr.pkl"
	# pkl_file = "result/denovo_qedlogpjnkgsk_start_ncncccn.pkl"
	pkl_file = "result/denovo_"+prop+".pkl"
	idx_2_smiles2f, trace_dict = pickle.load(open(pkl_file, 'rb'))
	bestvalue, best_smiles = 0, ''
	topk = 20
	whole_smiles2f = dict()
	for idx, (smiles2f,current_set) in tqdm(idx_2_smiles2f.items()):
		# if idx < 115:
		# 	continue 
		whole_smiles2f.update(smiles2f)
		for smiles,f in smiles2f.items():
			if f > bestvalue:
				bestvalue = f
				print("best", f)
				best_smiles = smiles 

	smiles_f_lst = [(smiles,f) for smiles,f in whole_smiles2f.items()]
	smiles_f_lst.sort(key=lambda x:x[1], reverse=True)
	best_smiles_lst = [smiles for smiles,f in smiles_f_lst[:topk]]
	best_f_lst = [f for smiles,f in smiles_f_lst[:topk]]
	avg, std = np.mean(best_f_lst), np.std(best_f_lst)
	print('f scores', str(avg)[:5], str(std)[:5])
	#### evaluate novelty

	t1 = time()
	nov = novelty(best_smiles_lst, zinc_lst)
	t2 = time()
	print("novelty", nov, "takes", str(int(t2-t1)), 'seconds')

	### evaluate diversity 
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

	for ii,smiles in enumerate(best_smiles_lst[:20]):
		print(smiles, str(jnk(smiles)))
		draw_smiles(smiles, "figure/best_"+prop+"_"+str(ii)+'.png')


# def draw_smiles(smiles, figfile_name):



'''
./download.sh result/denovo_jnk.pkl 
mv denovo_jnk.pkl result 
'''


'''
Clc1ccc(-c2ccn(-c3ccc(-c4ccnc(Nc5ccc(-n6cnc(N=S7C=CC=C7n7ccc(-c8cccs8)n7)n6)cc5)n4)cc3)n2)s1 0.97
Clc1ccc(-c2ccn(-c3ccc(-c4ccnc(Nc5ccc(-n6cnc(N=S7C=CC=C7n7ccc(-c8sccc8-n8cccn8)n7)n6)cc5)n4)cc3)n2)s1 0.97

'''

