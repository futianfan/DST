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
from chemutils import * 
'''
chemutils 
	smiles2differentiable_graph
	differentiable_graph2smiles
	qed_logp_jnk_gsk_fusion
'''
from tdc import Evaluator




## 2. data and oracle
start_smiles_lst = ['CO']

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
model_ckpt = "save_model/qed_logp_jnk_gsk_epoch_4_iter_14000_validloss_9715.ckpt"
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


'''



def distribution_learning(start_smiles_lst, gnn, oracle, generations, population_size):
	trace_dict = dict() 
	existing_set = set(start_smiles_lst)
	best_smiles_score_lst = []
	current_set = set(start_smiles_lst)
	for i_gen in tqdm(range(generations)):
		next_set = set()
		for smiles in current_set:
			smiles_set = optimize_single_molecule_one_iterate(smiles, gnn)
			for smi in smiles_set:
				if smi not in trace_dict:
					trace_dict[smi] = smiles 
			next_set = next_set.union(smiles_set)
		next_set = next_set.difference(existing_set)
		smiles_score_lst = oracle_screening(next_set, oracle)
		best_smiles_score_lst.extend(smiles_score_lst[:population_size])
		current_set = [i[0] for i in smiles_score_lst[:population_size]]		
		existing_set = existing_set.union(next_set)
		best_smiles_score_lst.sort(key=lambda x:x[1], reverse=True)
		print("best at", i_gen, "iter:", best_smiles_score_lst[:5])

	best_smiles_score_lst.sort(key=lambda x:x[1], reverse=True)
	return best_smiles_score_lst, existing_set, trace_dict  





## 5. run 
if __name__ == "__main__":
	generations = 10 
	population_size = 10
	result_file = "result/denovo_from_" + start_smiles_lst[0] + "_generation_" + str(generations) + "_population_" + str(population_size) + ".pkl"
	if not os.path.exists(result_file):
		best_smiles_score_lst, existing_set, trace_dict = distribution_learning(start_smiles_lst, gnn, oracle, 
																	generations = generations, population_size = population_size) 
		pickle.dump((best_smiles_score_lst,trace_dict), open(result_file, 'wb'))
	else:
		best_smiles_score_lst, trace_dict = pickle.load(open(result_file, 'rb'))

	best_smiles_lst = [smiles for smiles, score in best_smiles_score_lst]



	## traceback chain 
	from chemutils import draw_smiles
	smiles = best_smiles_lst[0]
	chain = [smiles]
	trace_dict.pop(start_smiles_lst[0])
	while chain[-1] in trace_dict:
		smiles = chain[-1]
		smiles = trace_dict[smiles]
		chain.append(smiles)
	print(' -> '.join(chain))
	f_lst = []
	for i,smiles in enumerate(chain):
		f_value = oracle(smiles)
		f_lst.append(f_value)
	f_lst = f_lst[-6:]
	f_lst = f_lst[::-1]
	plt.plot(f_lst)
	plt.xlabel("number of substructures")
	plt.ylabel("objective function")
	plt.savefig("figure/landscape.png")
	# print(smiles, str(qed(smiles))[:6], str(logp(smiles))[:6], str(jnk(smiles))[:6], str(gsk(smiles))[:6])





