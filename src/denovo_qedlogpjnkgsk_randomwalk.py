'''
ablation study 
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
import random 
random.seed(1)
from chemutils import * 
'''
chemutils 
	smiles2differentiable_graph
	differentiable_graph2smiles
	qed_logp_jnk_gsk_fusion
'''
from tdc import Evaluator




## 2. data and oracle
start_smiles_lst = ['C1=CC=CC=C1N']

global num_oracle_call
num_oracle_call = 0 
qed = Oracle(name = 'qed')
logp = Oracle(name = 'logp')
jnk = Oracle(name = 'JNK3')
gsk = Oracle(name = 'GSK3B')
def oracle(smiles):
	global num_oracle_call
	num_oracle_call += 1 
	scores = qed(smiles), logp(smiles), jnk(smiles), gsk(smiles)
	return qed_logp_jnk_gsk_fusion(*scores)



## 3. load model 
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu' ## cpu is better 
prop = 'qed_logp_jnk_gsk'
model_ckpt = "save_model0/qed_logp_jnk_gsk_epoch_4_iter_14000_validloss_9715.ckpt"
gnn = torch.load(model_ckpt)
gnn.switch_device(device)



## 4. inference function 
from inference_utils import * 

'''
def optimize_single_molecule_one_iterate(smiles, gnn):
	...
	return smiles_set


def gnn_screening(smiles_set, gnn):
	... 
	return smiles_lst

def oracle_screening(smiles_set, oracle):
	... 
	return smiles_score_lst 


def dpp(smiles_score_lst, num_return):

	return smiles_lst 

'''



def distribution_learning(start_smiles_lst, gnn, oracle, generations, population_size, lamb, result_pkl, random_topology, random_substr):
	trace_dict = dict() 
	existing_set = set(start_smiles_lst)  
	current_set = set(start_smiles_lst)
	average_f = np.mean([oracle(smiles) for smiles in current_set])
	f_lst = [(average_f, 0.0)]
	idx_2_smiles2f = {}
	smiles2f_new = {smiles:oracle(smiles) for smiles in start_smiles_lst} 
	idx_2_smiles2f[-1] = smiles2f_new, current_set 
	for i_gen in tqdm(range(generations)):
		next_set = set()
		for smiles in current_set:
			# smiles_set = optimize_single_molecule_one_iterate(smiles, gnn)  ### 
			if substr_num(smiles) < 3: #### short smiles
				smiles_set = optimize_single_molecule_one_iterate(smiles, gnn)  ### optimize_single_molecule_one_iterate_v2
			else:
				smiles_set = random_optimizing_single_moleccule_one_iteration(smiles, gnn, random_topology, random_substr)
			for smi in smiles_set:
				if smi not in trace_dict:
					trace_dict[smi] = smiles 
			next_set = next_set.union(smiles_set)
		# next_set = next_set.difference(existing_set)   ### if allow repeat molecule  
		smiles_score_lst = oracle_screening(next_set, oracle)  ###  sorted smiles_score_lst 
		# Option I:  select top-k molecule 
		# current_set = [i[0] for i in smiles_score_lst[:population_size]]  #
		# Option II: DPP
		current_set, log_det_v, log_det_s = dpp(smiles_score_lst = smiles_score_lst, num_return = population_size, lamb = lamb)
		print('len', len(smiles_score_lst), 'log_det_v', log_det_v, 'log_det_s', log_det_s)
		existing_set = existing_set.union(next_set)

		# save 
		smiles2f_new = {smiles:score for smiles,score in smiles_score_lst} 
		idx_2_smiles2f[i_gen] = smiles2f_new, current_set 
		pickle.dump((idx_2_smiles2f, trace_dict), open(result_pkl, 'wb'))

		#### compute f-score
		score_lst = [smiles2f_new[smiles] for smiles in current_set] 
		average_f = np.mean(score_lst)
		std_f = np.std(score_lst)
		f_lst.append((average_f, std_f))
		str_f_lst = [str(i[0])[:5]+'\t'+str(i[1])[:5] for i in f_lst]
		with open("result/denovo_" + prop + "_f_t.txt", 'w') as fout:
			fout.write('\n'.join(str_f_lst))




## 5. run 
if __name__ == "__main__":
	generations = 100
	population_size = 1
	## True False 
	### T,T    T,F     F,T     F,F
	import sys 
	random_topology = sys.argv[1]=='1'
	random_substr = sys.argv[2]=='1' 
	random_topology = True   
	random_substr = False  
	str_topo = 'random' if random_topology else 'dmg'
	str_substr = 'random' if random_substr else 'dmg'
	# result_file = "result/denovo_from_" + start_smiles_lst[0] + "_generation_" + str(generations) + "_population_" + str(population_size) + ".pkl"
	result_pkl = "result/ablation_" + str_topo + "_topo_" + str_substr + "_substr.pkl"
	distribution_learning(start_smiles_lst, gnn, oracle, 
										generations = generations, 
										population_size = population_size, 
										lamb=2, 
										result_pkl = result_pkl, 
										random_topology = random_topology, 
										random_substr = random_substr) 










