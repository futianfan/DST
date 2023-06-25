#### user input ####
objectives = ['GSK3',]
oracle_budget = 5000
pretrain_budget = 2000
#### user input ####




from tdc import Oracle 
oracle_list = [Oracle(objective) for objective in objectives]
oracle_dic_list = [dict() for objective in objectives]
objectives2idx = {objective:idx for idx,objective in enumerate(objectives)}

from tqdm import tqdm 
import os
import numpy as np 
from random import shuffle 
from tdc.generation import MolGen
from tdc import Evaluator 
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import pickle 
torch.manual_seed(4) 
np.random.seed(2)
from module import GCN 
from chemutils import smiles2graph, vocabulary 
from utils import Molecule_Dataset 
from time import time
from chemutils import * 
from inference_utils import * 
device = 'cpu'

########################## 1. data ##########################
data = MolGen(name = 'ZINC')
from chemutils import is_valid, logp_modifier 
smiles_database = "data/zinc.tab"
clean_smiles_database = "data/zinc_clean.txt"


with open(smiles_database, 'r') as fin:
	lines = fin.readlines()[1:]
smiles_lst = [i.strip().strip('"') for i in lines]
shuffle(smiles_lst)
smiles_lst = smiles_lst[:pretrain_budget]
clean_smiles_lst = []
for smiles in tqdm(smiles_lst):
	if is_valid(smiles):
		clean_smiles_lst.append(smiles)
clean_smiles_set = set(clean_smiles_lst)
with open(clean_smiles_database, 'w') as fout:
	for smiles in clean_smiles_set:
		fout.write(smiles + '\n')
clean_smiles_lst = list(clean_smiles_set)

def collate_fn(batch_lst):
	return [element[0] for element in batch_lst], [element[1] for element in batch_lst]


########################## 2. pretrain model ########################## 
gnn_list = [GCN(nfeat = 50, nhid = 100, n_out = 1, num_layer = 3) for i in oracle_list]

for idx, (gnn, oracle, objective) in enumerate(zip(gnn_list, oracle_list, objectives)):
	#### training data ####  
	lines = [(smiles, oracle(smiles)) for smiles in clean_smiles_lst]
	N = int(len(lines) * 0.9)
	train_data = lines[:N]
	valid_data = lines[N:] 

	training_set = Molecule_Dataset(train_data)
	valid_set = Molecule_Dataset(valid_data)
	params = {'batch_size': 1,
	          'shuffle': True,
	          'num_workers': 1}

	train_generator = torch.utils.data.DataLoader(training_set, collate_fn = collate_fn, **params)
	valid_generator = torch.utils.data.DataLoader(valid_set, collate_fn = collate_fn, **params)

	cost_lst = []
	valid_loss_lst = []
	epoch = 1 
	every_k_iters = 1000
	save_folder = "save_model/" + objective + "_epoch_" 
	for ep in tqdm(range(epoch)):
		for i, (smiles, score) in tqdm(enumerate(train_generator)):
			### 1. training
			smiles = smiles[0]
			y = torch.FloatTensor(score)
			idx_lst, node_mat, substructure_lst, atomidx_2substridx, adjacency_matrix, leaf_extend_idx_pair = smiles2graph(smiles)
			idx_vec = torch.LongTensor(idx_lst).to(device)
			node_mat = torch.FloatTensor(node_mat).to(device)
			adjacency_matrix = torch.FloatTensor(adjacency_matrix).to(device)
			weight = torch.ones_like(idx_vec).to(device)
			cost = gnn.learn(node_mat, adjacency_matrix, weight, y)
			cost_lst.append(cost)

			#### 2. validation 
			if i % every_k_iters == 0:
				gnn.eval()
				valid_loss, valid_num = 0,0 
				for smiles,score in valid_generator:
					smiles = smiles[0]
					y = torch.FloatTensor(score).to(device)
					idx_lst, node_mat, substructure_lst, atomidx_2substridx, adjacency_matrix, leaf_extend_idx_pair = smiles2graph(smiles)
					idx_vec = torch.LongTensor(idx_lst).to(device)
					node_mat = torch.FloatTensor(node_mat).to(device)
					adjacency_matrix = torch.FloatTensor(adjacency_matrix).to(device)
					weight = torch.ones_like(idx_vec).to(device)
					cost, _ = gnn.valid(node_mat, adjacency_matrix, weight, y)
					valid_loss += cost
					valid_num += 1 
				valid_loss = valid_loss / valid_num
				valid_loss_lst.append(valid_loss)
				file_name = save_folder + str(ep) + "_iter_" + str(i) + "_validloss_" + str(valid_loss)[:7] + ".ckpt"
				torch.save(gnn, file_name)
				gnn.train()
########################## 3. de novo design ########################## 
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def multi_optimization(start_smiles_lst, gnn_list, oracle_list, oracle_num, generations, population_size, lamb, topk, epsilon, result_pkl):
	smiles2score = dict() ### oracle_num
	def oracle_new(smiles):
		if smiles not in smiles2score:
			value = np.mean([oracle(smiles) for oracle in oracle_list])
			smiles2score[smiles] = value 
		return smiles2score[smiles] 
	trace_dict = dict() 
	existing_set = set(start_smiles_lst)  
	current_set = set(start_smiles_lst)
	average_f = np.mean([oracle_new(smiles) for smiles in current_set])
	f_lst = [(average_f, 0.0)]
	idx_2_smiles2f = {}
	smiles2f_new = {smiles:oracle_new(smiles) for smiles in start_smiles_lst} 
	idx_2_smiles2f[-1] = smiles2f_new, current_set 
	for i_gen in tqdm(range(generations)):
		next_set = set()
		for smiles in current_set:
			# smiles_set = optimize_single_molecule_one_iterate(smiles, gnn)  ### 
			if substr_num(smiles) < 3: #### short smiles
				smiles_set = optimize_single_molecule_one_iterate_gnnlist(smiles, gnn_list)  ### optimize_single_molecule_one_iterate_v2
			else:
				smiles_set = optimize_single_molecule_one_iterate_v3_gnnlist(smiles, gnn_list, topk = topk, epsilon = epsilon)
			for smi in smiles_set:
				if smi not in trace_dict:
					trace_dict[smi] = smiles 
			next_set = next_set.union(smiles_set)
		# next_set = next_set.difference(existing_set)   ### if allow repeat molecule  
		smiles_score_lst = oracle_screening(next_set, oracle_new)  ###  sorted smiles_score_lst 
		print(smiles_score_lst[:5], "Oracle num", len(smiles2score))

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
		with open("result/" + "f_t.txt", 'w') as fout:
			fout.write('\n'.join(str_f_lst))
		if len(smiles2score) > oracle_num: 
			break 


start_smiles_lst = ['C1(N)=NC=CC=N1']
generations = 50
population_size = 20
oracle_num = oracle_budget
result_pkl = "result/" + "oracle.pkl"
multi_optimization(start_smiles_lst, gnn_list, oracle_list, oracle_num,
						generations = generations, 
						population_size = population_size, 
						lamb=2, 
						topk = 5, 
						epsilon = 0.7, 
						result_pkl = result_pkl) 









