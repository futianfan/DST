'''
	1. import 
	2. config 
		device
		data 
		hyperparameter 

	3. data loader 
	4. model 
	5. learn & valid 

'''

## 1. import 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from tqdm import tqdm 
from matplotlib import pyplot as plt
import pickle 
from random import shuffle 
torch.manual_seed(4) 
np.random.seed(2)
from module import GCN 
from chemutils import smiles2graph, vocabulary 
from utils import Molecule_Dataset 


# ['JNK3', 'GSK3B', 'DRD2']
prop = 'qedsajnkgsk'




## 2. config
## 2.1 device
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

## 2.2 data 
data_file = "data/" + prop + ".txt"
with open(data_file, 'r') as fin:
	lines = fin.readlines() 
lines = [(line.split()[0], float(line.split()[1])) for line in lines]
shuffle(lines)
N = int(len(lines) * 0.9)
train_data = lines[:N]
valid_data = lines[N:]






## 3. data loader 
training_set = Molecule_Dataset(train_data)
valid_set = Molecule_Dataset(valid_data)
params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 1}

def collate_fn(batch_lst):
	return [element[0] for element in batch_lst], [element[1] for element in batch_lst]

train_generator = torch.utils.data.DataLoader(training_set, collate_fn = collate_fn, **params)
valid_generator = torch.utils.data.DataLoader(valid_set, collate_fn = collate_fn, **params)
print('data loader is built!')








## 4. model 
gnn = GCN(nfeat = 50, nhid = 100, n_out = 1, num_layer = 3).to(device)
print('GNN is built!')




## 5. learn 
'''
	chemutils.smiles2differentiable_graph 
	chemutils.smiles2graph 

	&&  

	module.GCN.forward 

'''

cost_lst = []
valid_loss_lst = []
epoch = 20 
every_k_iters = 10000
save_folder = "save_model/" + prop + "_epoch_" 
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
		# if i % every_k_iters == 0 and i > 0:
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
			# plt.cla()
			# plt.plot(valid_loss_lst)
			# plt.savefig("figure/" + prop + "_valid_loss.png")
			file_name = save_folder + str(ep) + "_iter_" + str(i) + "_validloss_" + str(valid_loss)[:7] + ".ckpt"
			torch.save(gnn, file_name)
			gnn.train()










