import numpy as np 
import matplotlib.pyplot as plt
import pickle 
from random import shuffle 
import matplotlib.cm as cm
import torch 
from tqdm import tqdm
from tdc import Oracle
prop = 'qed'
oracle = Oracle(name = prop)
from chemutils import  is_valid

model_ckpt = "save_model/QED_epoch_0_iter_75900_validloss_0.5631.ckpt"
# model_ckpt = "save_model0/qed_logp_jnk_gsk_epoch_4_iter_14000_validloss_9715.ckpt"
gnn = torch.load(model_ckpt)
def gnn_pred(smiles):
	return gnn.smiles2pred(smiles)
pkl_file = "result/denovo_from_NC1=NC=CC=N1_qed.pkl"
idx_2_smiles2f, trace_dict = pickle.load(open(pkl_file, 'rb'))
generated_smiles_set = set()
idx2stat = {}
for idx,x in tqdm(idx_2_smiles2f.items()):
	# if idx > 50:
	# 	continue 
	smiles2f, current_set = x 
	current_set = list(current_set)
	# current_f = [smiles2f[smiles] for smiles in current_set]
	current_f = list(smiles2f.values())
	gnn_pred_list = []
	for smiles in current_set:
		if is_valid(smiles):
			gnn_pred_list.append(gnn_pred(smiles))
	scores = oracle(current_set)
	idx2stat[idx] = np.mean(current_f), np.std(current_f), \
					np.mean(gnn_pred_list), np.std(gnn_pred_list), \
					np.mean(scores), np.std(scores)

sort_idx_lst = list(idx_2_smiles2f.keys())
sort_idx_lst.sort()
# sort_idx_lst = sort_idx_lst[:50]
sort_stats = [idx2stat[idx] for idx in sort_idx_lst]

labels = ['f', 'gnn', prop, ]
colors = ['r', 'b', 'g', 'y', 'c', 'm']
for i in range(3):
	avg_list = [stat[i*2] for stat in sort_stats]
	std_list = [stat[i*2+1] for stat in sort_stats] ### 1.96 -- 95% confidence interval
	r1 = [i-j for i,j in zip(avg_list, std_list)]
	r2 = [i+j for i,j in zip(avg_list, std_list)]
	color = cm.viridis(0.7)
	plt.plot(list(range(len(avg_list))), avg_list, label = labels[i], color = colors[i])
	plt.fill_between(list(range(len(avg_list))), r1, r2, color=colors[i], alpha=0.2, )

	# if i==1:
	# 	plt.legend(fontsize=18, loc=1)
	# 	plt.tight_layout()
	# 	plt.savefig('figure/' + prop + '_fscore.png')
	# 	plt.cla()



plt.legend(fontsize=18, loc=1)
plt.tight_layout()
plt.savefig('figure/' + prop + '_raw_score.png')
plt.cla()


'''

result/ablation_dmg_topo_dmg_substr.pkl
result/ablation_dmg_topo_random_substr.pkl
result/ablation_random_topo_dmg_substr.pkl
result/ablation_random_topo_random_substr.pkl


'''


