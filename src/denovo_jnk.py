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
from tdc import Evaluator


start_smiles_lst = ['C1(N)=NC=CC=N1']
## 'C1=CC=CC=C1NC2=NC=CC=N2'
jnk = Oracle(name = 'JNK3')
def oracle(smiles):
	return jnk(smiles)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu' ## cpu is better 
prop = 'jnk'
# model_ckpt = "save_model/JNK3_epoch_0_iter_93000_validloss_0.0772.ckpt"
model_ckpt = "save_model0/qed_logp_jnk_gsk_epoch_4_iter_14000_validloss_9715.ckpt"
gnn = torch.load(model_ckpt)
gnn.switch_device(device)



## 4. inference function 
from inference_utils import * 





def distribution_learning(start_smiles_lst, gnn, oracle, generations, population_size, lamb, topk, epsilon, result_pkl):
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
				smiles_set = optimize_single_molecule_one_iterate_v3(smiles, gnn, topk = topk, epsilon = epsilon)
			for smi in smiles_set:
				if smi not in trace_dict:
					trace_dict[smi] = smiles 
			next_set = next_set.union(smiles_set)
		smiles_score_lst = oracle_screening(next_set, oracle)  ###  sorted smiles_score_lst 
		print(smiles_score_lst[:7])

		# current_set = [i[0] for i in smiles_score_lst[:population_size]]  # Option I: top-k 
		current_set,_,_ = dpp(smiles_score_lst = smiles_score_lst, num_return = population_size, lamb = lamb) 	# Option II: DPP
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
	generations = 300
	population_size = 10
	result_pkl = "result/denovo_" + prop + ".pkl"
	distribution_learning(start_smiles_lst, gnn, oracle, 
							generations = generations, 
							population_size = population_size, 
							lamb=2, 
							topk = 5, 
							epsilon = 0.7, 
							result_pkl = result_pkl) 










