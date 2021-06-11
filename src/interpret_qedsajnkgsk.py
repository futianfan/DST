import numpy as np 
import matplotlib.pyplot as plt
import pickle 
from random import shuffle 
import matplotlib.cm as cm
import torch 
from tqdm import tqdm
from tdc import Oracle
from collections import defaultdict
prop = 'logp'
oracle = Oracle(name = prop)
from chemutils import * 
# is_valid, substr_num smiles2differentiable_graph_v2 
def sigmoid(x):
	return 1/(1+np.exp(-x))

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = [
    'Roboto Condensed', 'Roboto Condensed Regular'
]

from copy import deepcopy
from rdkit import Chem, DataStructs
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem, Draw, Descriptors, QED
import io

from PIL import Image


def mol_with_atom_index(mol):
    mol_ = deepcopy(mol)
    for atom in mol_.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol_

def show_png(data):
    bio = io.BytesIO(data)
    img = Image.open(bio)
    return img

# def value2color(v):
#     if v >=0:
#         return (1.0, 1-v, 1-v)
#     else:
#         v = -v
#         return (1-v, 1-v, 1.0)


def value2color(v):
    v = v * 30
    if v >=0:
        v = min(v,1.0)
        return (1.0, 1-v, 1-v)
    else:
        v = -v
        v = min(v,1.0)
        return (1-v, 1-v, 1.0)

vocabulary = load_vocabulary()
model_ckpt = "qedsajnkgsk_epoch_15_iter_50000_validloss_0.62882.ckpt"
gnn = torch.load(model_ckpt)
def gnn_pred(smiles):
	return gnn.smiles2pred(smiles)
pkl_file = "result/denovo_from_CC_" + prop + ".pkl"
idx_2_smiles2f, trace_dict = pickle.load(open(pkl_file, 'rb'))
generated_smiles_set = set()
idx2stat = {}
# for ii, (smiles, ancestor_smiles) in enumerate(trace_dict.items()): 


	# if substr_num(smiles) != 3:
	# 	continue 
	# if len(smiles) < len(ancestor_smiles) + 7:
	# 	continue 
ancestor_smiles = "Nc1ccc(-c2ccnc(Nc3ccc(-n4cncn4)cc3)n2)cc1"

if True:
	diff_graph = smiles2differentiable_graph_v3(ancestor_smiles) 
	(is_nonleaf, is_leaf, is_extend), node_indicator, adjacency_mask, \
	adjacency_weight, leaf_extend_idx_pair, leaf_nonleaf_lst, atomidx_2substridx = diff_graph
	substridx2atoms = defaultdict(lambda:[])
	for atom_idx, substr_idx in atomidx_2substridx.items():
		substridx2atoms[substr_idx].append(atom_idx)
	# print(ancestor_smiles, '->', smiles)
	node_mask = is_nonleaf 
	node_indicator_np2, adjacency_weight_np2, node_indicator_grad, adjacency_weight_grad = gnn.update_molecule_interpret(node_mask, node_indicator, adjacency_mask, adjacency_weight)
	leaf2nonleaf = {leaf:nonleaf for leaf,nonleaf in leaf_nonleaf_lst}
	for leaf_idx, extend_idx in leaf_extend_idx_pair:
		# for substr, prob in zip(vocabulary, leaf_substr):
		# 	print('\t', substr, prob)
		extend_substr = node_indicator_np2[extend_idx,:]
		extend_prob = np.exp(extend_substr)
		extend_prob = extend_prob / np.sum(extend_prob)
		sorted_extend_prob = np.argsort(extend_prob)
		sorted_extend_prob = list(sorted_extend_prob)
		sorted_extend_prob = sorted_extend_prob[::-1]
		leaf_weight = set(list(adjacency_weight_np2[leaf_idx,:]))
		leaf_weight.remove(adjacency_weight_np2[leaf_idx,extend_idx])
		leaf_weight1 = max(list(leaf_weight))
		leaf_weight = set(list(adjacency_weight_np2[:,leaf_idx]))
		leaf_weight.remove(adjacency_weight_np2[extend_idx, leaf_idx])
		leaf_weight2 = max(list(leaf_weight))
		leaf_weight = (sigmoid(leaf_weight1) + sigmoid(leaf_weight2)) / 2 
		# print("leaf weight", leaf_weight)
		extend_weight = (sigmoid(adjacency_weight_np2[leaf_idx,extend_idx]) + sigmoid(adjacency_weight_np2[extend_idx,leaf_idx]))/2
		# print("extend weight", extend_weight)
		leaf_substr = node_indicator_np2[leaf_idx,:]
		# print("----leaf substr -----")
		# print("====extend substr =====")
		for idx in sorted_extend_prob[:25]:
			print('\t', vocabulary[idx], extend_prob[idx])
		print("leaf_weight", leaf_weight)
		print("extend weight", extend_weight)
		nonleaf = leaf2nonleaf[leaf_idx]
		print("leaf nonleaf gradient", adjacency_weight_grad[leaf_idx, nonleaf]+ adjacency_weight_grad[nonleaf, leaf_idx])
		print("leaf extend gradient", adjacency_weight_grad[leaf_idx, extend_idx]+ adjacency_weight_grad[extend_idx, leaf_idx])
		# for substr, prob in zip(vocabulary, extend_substr):
		# 	print('\t', substr, prob)
	### visualize molecule
	# atom2value = {}
	# leaf2nonleaf = {leaf:nonleaf for leaf,nonleaf in leaf_nonleaf_lst}
	# leaf2extend = {leaf:extends for leaf,extends in leaf_extend_idx_pair}
	# for leaf in leaf2extend:
	# 	leaf_atom_idx_lst = substridx2atoms[leaf]
	# 	nonleaf = leaf2nonleaf[leaf]
	# 	leaf_weight = adjacency_weight_grad[leaf, nonleaf] + adjacency_weight_grad[nonleaf, leaf]
	# 	for atom in leaf_atom_idx_lst: 
	# 		atom2value[atom] = leaf_weight 

	# print("atom2value", atom2value)
	# # atom2value = {0:0.5, 1:0.3, 2:0.9}
	# mol = Chem.MolFromSmiles(ancestor_smiles)
	# atom2color = {atom:value2color(value) for atom,value in atom2value.items()}
	# atom_lst = [atom for atom in atom2color]
	# print("atom_lst", atom_lst, atom2color)
	# d = rdMolDraw2D.MolDraw2DCairo(500, 500)
	# rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=atom_lst,
 #                                   highlightAtomColors=atom2color,)

	# d.FinishDrawing()
	# img = show_png(d.GetDrawingText())
	# print("figure/color_"+str(ii)+".png")
	# img.save("figure/color_"+str(ii)+".png")






