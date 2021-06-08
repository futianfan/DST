from tqdm import tqdm 
import os
# from chemutils import vocabulary, smiles2word 
from chemutils import is_valid, logp_modifier 
smiles_database = "raw_data/zinc.tab"
clean_smiles_database = "data/clean_zinc.txt"


with open(smiles_database, 'r') as fin:
	lines = fin.readlines()[1:]
smiles_lst = [i.strip().strip('"') for i in lines]

clean_smiles_lst = []
for smiles in tqdm(smiles_lst):
	if is_valid(smiles):
		clean_smiles_lst.append(smiles)
clean_smiles_set = set(clean_smiles_lst)





# for prop in ['JNK3', 'GSK3B', 'DRD2', 'QED']:
# 	property_mol_file = "data/zinc_" + prop + ".txt"
# 	clean_property_mol_file = "data/zinc_" + prop + "_clean.txt"
# 	if not os.path.exists(clean_property_mol_file):
# 		with open(property_mol_file, 'r') as fin, open(clean_property_mol_file, 'w') as fout:
# 			lines = fin.readlines()
# 			for line in tqdm(lines):
# 				smiles = line.split()[0]
# 				if smiles in clean_smiles_set:
# 					fout.write(line)


# for prop in ['LogP']:
# 	property_mol_file = "data/zinc_" + prop + ".txt"
# 	clean_property_mol_file = "data/zinc_" + prop + "_clean.txt"
# 	if not os.path.exists(clean_property_mol_file):
# 		with open(property_mol_file, 'r') as fin, open(clean_property_mol_file, 'w') as fout:
# 			lines = fin.readlines()
# 			for line in tqdm(lines):
# 				smiles = line.split()[0]
# 				score = float(line.split()[1])
# 				score2 = logp_modifier(score)
# 				if smiles in clean_smiles_set:
# 					fout.write(smiles + '\t' + str(score2) + '\n')


# def logp_modifier2(logp_score):
#     return max(0.0,min(1.0,1/40*(logp_score+20))) 


for prop in ['LogP']:
	property_mol_file = "data/zinc_" + prop + ".txt"
	clean_property_mol_file = "data/zinc_" + prop + "_clean_raw.txt"
	if not os.path.exists(clean_property_mol_file):
		with open(property_mol_file, 'r') as fin, open(clean_property_mol_file, 'w') as fout:
			lines = fin.readlines()
			for line in tqdm(lines):
				smiles = line.split()[0]
				score = float(line.split()[1])
				# score2 = logp_modifier2(score)
				if smiles in clean_smiles_set:
					fout.write(smiles + '\t' + str(score) + '\n')



prop = 'SA'
# property_mol_file = "data/zinc_" + prop + ".txt"
# 	clean_property_mol_file = "data/zinc_" + prop + "_clean2.txt"
# 	if not os.path.exists(clean_property_mol_file):
# 		with open(property_mol_file, 'r') as fin, open(clean_property_mol_file, 'w') as fout:
# 			lines = fin.readlines()
# 			for line in tqdm(lines):
# 				smiles = line.split()[0]
# 				score = float(line.split()[1])
# 				score2 = logp_modifier2(score)
# 				if smiles in clean_smiles_set:
# 					fout.write(smiles + '\t' + str(score2) + '\n')












