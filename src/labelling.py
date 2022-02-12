from tqdm import tqdm 
import numpy as np 

from tdc import Oracle


rawdata_file = "raw_data/zinc.tab"
with open(rawdata_file) as fin:
	lines = fin.readlines()[1:]
	smiles_lst = [line.strip().strip('"') for line in lines]

oracle_names = ['LogP']
# oracle_names = ['SA']
## GSK3B, JNK3, QED, DRD2, LogP, SA

batch_size = 10
num_of_batch = int(np.ceil(len(smiles_lst) / batch_size))


# for oracle in oracle_names:
# 	print(oracle)
# 	output_file = "data/zinc_" + str(oracle) + '.txt'
# 	oracle = Oracle(name = oracle)
# 	with open(output_file, 'w') as fout:
# 		for i in tqdm(range(num_of_batch)):
# 			start_idx = i*batch_size 
# 			end_idx = i*batch_size + batch_size 
# 			sub_smiles_lst = smiles_lst[start_idx:end_idx] 
# 			score_lst = oracle(sub_smiles_lst) 
# 			for smiles,score in zip(sub_smiles_lst, score_lst):
# 				fout.write(smiles + '\t' + str(score) + '\n') 





sa = Oracle(name = 'SA')
mu = 2.230044
sigma = 0.6526308
def sa_oracle(smiles):
	sa_score = sa(smiles)
	mod_score = np.maximum(sa_score, mu)
	return np.exp(-0.5 * np.power((mod_score - mu) / sigma, 2.)) 

def sa_oracle_lst(smiles_lst):
	return [sa_oracle(smiles) for smiles in smiles_lst]

with open("data/clean_zinc.txt", 'r') as fin:
	lines = fin.readlines()
print(len(lines))
smiles_lst = [line.strip() for line in lines]
output_file = "data/zinc_sa_clean.txt"
batch_size = 10
num_of_batch = int(np.ceil(len(smiles_lst) / batch_size))

with open(output_file, 'w') as fout:
	for i in tqdm(range(num_of_batch)):
		start_idx = i*batch_size 
		end_idx = i*batch_size + batch_size 
		sub_smiles_lst = smiles_lst[start_idx:end_idx] 
		score_lst = sa_oracle_lst(sub_smiles_lst) 
		for smiles,score in zip(sub_smiles_lst, score_lst):
			fout.write(smiles + '\t' + str(score) + '\n') 




'''

ZINC 250K 

	QED  6 min 

	LogP <1.5hours 

	DRD2 24 min

	JNK3 10 hours 
		----- 0.15 second/mol

	GSK 10 hours   
		----- 0.15 second/mol


ChemBL 1M

	QED  xxx

	LogP xxx

	DRD2 xxx

	JNK3 xxx hours 
		-----  second/mol

	GSK xxx hours   
		----- xxx second/mol




'''


