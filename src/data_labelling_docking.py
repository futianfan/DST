from tqdm import tqdm 
import os
import numpy as np 
from chemutils import docking_modifier

### clean smiles set 
clean_smiles_database = "data/clean_zinc.txt"
with open(clean_smiles_database, 'r') as fin:
	lines = fin.readlines() 
clean_smiles_set = set([line.strip() for line in lines])


## raw docking
datafile = "data/raw_docking.txt"
with open(datafile, 'r') as fin:
	lines = fin.readlines() 
smiles2score = {line.split()[0]:float(line.split()[1]) for line in lines}
smiles_score_lst = [(smiles, score) for smiles, score in smiles2score.items() if smiles in clean_smiles_set]



outputfile = "data/docking_drd3.txt"
with open(outputfile, 'w') as fout:
	for smiles, score in tqdm(smiles_score_lst):
		if smiles in clean_smiles_set:
			score = docking_modifier(score)
			fout.write(smiles + '\t' + str(score) + '\n')


