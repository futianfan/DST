from tqdm import tqdm 
import os
# from chemutils import vocabulary, smiles2word 
from chemutils import is_valid, jnk_gsk_fusion
import numpy as np 


### clean smiles set 
clean_smiles_database = "data/clean_zinc.txt"
with open(clean_smiles_database, 'r') as fin:
	lines = fin.readlines() 
clean_smiles_set = set([line.strip() for line in lines])



### mapping: smiles -> prop 
prop_lst = ['JNK3', 'GSK3B']
prop_smiles_score_dict = dict()
for prop in prop_lst:
	raw_data = "data/zinc_" + prop + ".txt"
	with open(raw_data, 'r') as fin:
		lines = fin.readlines()
	prop_smiles_score_dict[prop] = {line.split()[0]:float(line.split()[1]) for line in lines}



### write results 
output_file = "data/jnkgsk.txt"
with open(output_file, 'w') as fout:
	for smiles in tqdm(clean_smiles_set):
		score_lst = (prop_smiles_score_dict[prop][smiles] for prop in prop_lst)
		label = jnk_gsk_fusion(*score_lst)
		fout.write(smiles + '\t' + str(label) + '\n')





