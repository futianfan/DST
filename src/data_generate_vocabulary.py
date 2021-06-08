from chemutils import smiles2word

import os
from collections import defaultdict 
from tqdm import tqdm 

all_vocabulary_file = "data/all_vocabulary.txt"
rawdata_file = "raw_data/zinc.tab"
select_vocabulary_file = "data/selected_vocabulary.txt"

if not os.path.exists(all_vocabulary_file):
	with open(rawdata_file) as fin:
		lines = fin.readlines()[1:]
		smiles_lst = [line.strip().strip('"') for line in lines]
	word2cnt = defaultdict(int)
	for smiles in tqdm(smiles_lst):
		word_lst = smiles2word(smiles)
		for word in word_lst:
			word2cnt[word] += 1
	word_cnt_lst = [(word,cnt) for word,cnt in word2cnt.items()]
	word_cnt_lst = sorted(word_cnt_lst, key=lambda x:x[1], reverse = True)

	with open(all_vocabulary_file, 'w') as fout:
		for word, cnt in word_cnt_lst:
			fout.write(word + '\t' + str(cnt) + '\n')
else:
	with open(all_vocabulary_file, 'r') as fin:
		lines = fin.readlines()
		word_cnt_lst = [(line.split('\t')[0], int(line.split('\t')[1])) for line in lines]


word_cnt_lst = list(filter(lambda x:x[1]>1000, word_cnt_lst))
print(len(word_cnt_lst))

with open(select_vocabulary_file, 'w') as fout:
	for word, cnt in word_cnt_lst:
		fout.write(word + '\t' + str(cnt) + '\n')



