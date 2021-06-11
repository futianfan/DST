
### 1. import
import numpy as np 
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



## 2. data and oracle
datafile = "data/qed_logp_jnk_gsk.txt"
with open(datafile, 'r') as fin:
	lines = fin.readlines() 
smiles2score = {line.split()[0]:float(line.split()[1]) for line in lines}

good_smiles2score = {smiles:score for smiles,score in smiles2score.items() if score > 0.45}
# print(max(smiles2score.values()))
# print(len(good_smiles2score))
# exit()
smiles_lst = [smiles for smiles in good_smiles2score]
smiles_lst = smiles_lst[1:]
smiles_lst = ['C1=CC=CC=C1N']

qed = Oracle(name = 'qed')
logp = Oracle(name = 'logp')
jnk = Oracle(name = 'JNK3')
gsk = Oracle(name = 'GSK3B')
name2oracle = {'qed':qed, 'logp':logp, 'JNK3':jnk, 'GSK3B':gsk}
def oracle(smiles):
	scores = qed(smiles), logp(smiles), jnk(smiles), gsk(smiles)
	return qed_logp_jnk_gsk_fusion(*scores)

result_file = "result/example_qedlogpjnkgsk.txt"





## 3. load model 
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu' ## cpu is better 
model_ckpt = "save_model0/qed_logp_jnk_gsk_epoch_4_iter_14000_validloss_9715.ckpt"
gnn = torch.load(model_ckpt)
# exit()
gnn.switch_device(device)





## 4. inference function 
from inference_utils import optimize_single_molecule_all_generations

def calculate_results(input_smiles, input_score, result_file, best_mol_score_list, oracle):
	if best_mol_score_list == []:
		with open(result_file, 'a') as fout:
			fout.write("fail to optimize" + input_smiles + '\n')
		return None 
	output_scores = [i[1] for i in best_mol_score_list]
	smiles_lst = [i[0] for i in best_mol_score_list]
	with open(result_file, 'a') as fout:
		fout.write(str(input_score) + '\t' + str(output_scores[0]) + '\t' + str(np.mean(output_scores[:3]))
				 + '\t' + input_smiles + '\t' + ' '.join(smiles_lst[:3]) + '\n')
	return input_score, output_scores[0]


def inference_single_molecule(input_smiles, gnn, result_file, generations, population_size, lamb):
	best_mol_score_list, input_score, traceback_dict = optimize_single_molecule_all_generations(input_smiles, gnn, oracle, generations, population_size, lamb)
	return calculate_results(input_smiles, input_score, result_file, best_mol_score_list, oracle)


def inference_molecule_set(input_smiles_lst, gnn, result_file, generations, population_size, lamb):
	score_lst = []
	for input_smiles in tqdm(input_smiles_lst):
		if not is_valid(input_smiles):
			continue 
		result = inference_single_molecule(input_smiles, gnn, result_file, generations, population_size, lamb)
		if result is None:
			score_lst.append(None)
		else:
			input_score, output_score = result
			score_lst.append((input_score, output_score))
	return score_lst





## 5. run 
if __name__ == "__main__":
	score_lst = inference_molecule_set(smiles_lst, gnn, result_file = result_file, generations = 3, population_size = 10, lamb = 1)


'''

C1=C(c2cncn2-c2cc(Nc3ccc(N4CCOCC4)cc3)ncn2)NCC1 0.5642826199356552
C1=CC(c2cncn2-c2cc(Nc3cccc(-c4ccon4)c3)ncn2)CN1 0.5508714182384542



'''





