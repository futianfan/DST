
### 1. import
import numpy as np 
from tqdm import tqdm 
from matplotlib import pyplot as plt
import pickle , sys
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

qed = Oracle(name = 'qed')
logp = Oracle(name = 'logp')
jnk = Oracle(name = 'JNK3')
gsk = Oracle(name = 'GSK3B')
def oracle(smiles):
	scores = qed(smiles), logp(smiles), jnk(smiles), gsk(smiles)
	return qed_logp_jnk_gsk_fusion(*scores)




# smiles = 'C1=C(c2cncn2-c2cc(Nc3ccc(N4CCOCC4)cc3)ncn2)NCC1'
# smiles = 'C1=CC(c2cncn2-c2cc(Nc3cccc(-c4ccon4)c3)ncn2)CN1'
# smiles = 'C1=CN(C2C=C(c3cncn3-c3cc(Nc4ccc(N5CCOCC5)cc4)ncn3)NC2)CC1'
# smiles = 'Cc1cccc(Nc2nccc(C3CCCN3Cc3ccncc3C3COCCN3)n2)n1'
smiles = 'Cc1cccc(Nc2nccc(C3CCCN3Cc3ccncc3N3CCOCC3)n2)n1'
smiles = 'CCn1nc(C=C2C=CN=N2)c(-c2ccnc(NCc3cnn(C)c3)n2)c1C'
smiles = 'C1=CC=CC=C1NC2=NC=CC(C3=CC=CC=C3)=N2'
smiles = 'C1=CC=CC=C1NC2=NC=CC(C3=CC=C(F)C=C3)=N2'
smiles = 'C1=CN(n2cncn2)C(Nc2ccc(C3=CCCN3)cc2)C1'
smiles = 'C1=CC=CC=C1NC2=NC=CC=N2'
fig_name = 'figure/tmp.png'

draw_smiles(smiles, fig_name)
score_list = [oracle(smiles), qed(smiles), logp(smiles), jnk(smiles), gsk(smiles)]
score_list = [str(i)[:5] for i in score_list]
print(', '.join(score_list))



'''

C1=C(c2cncn2-c2cc(Nc3ccc(N4CCOCC4)cc3)ncn2)NCC1 0.564, 0.694, 0.348, 0.52, 0.38
C1=CC(c2cncn2-c2cc(Nc3cccc(-c4ccon4)c3)ncn2)CN1 0.5508714182384542
C1=CN(C2C=C(c3cncn3-c3cc(Nc4ccc(N5CCOCC5)cc4)ncn3)NC2)CC1 0.5600605735606905


Cc1cccc(Nc2nccc(C3CCCN3Cc3ccncc3C3COCCN3)n2)n1 0.571423456519113
Cc1ccnc(Nc2nccc(C3CCCN3Cc3ccncc3C3COCCN3)n2)c1 0.5687471246445096
Cc1cccc(Nc2nccc(C3CCCN3Cc3ccncc3N3CCOCC3)n2)n1 0.5552122165286687
Cc1ccnc(Nc2nccc(C3CCCN3Cc3ccncc3N3CCOCC3)n2)c1 0.5510679409769816


C1=CC=CC=C1NC2=NC=CC(C3=CC=CC=C3)=N2
C1=CC=CC=C1NC2=NC=CC(C3=CC=C(F)C=C3)=N2

init by "C1=CC=CC=C1N"

C1=CC=CC=C1
 ->
C1=CC=CC=C1N 
 ->
C1=CC=CC=C1NC2=NC=CC=N2
 -> 
C1=CC=CC=C1NC2=NC=CC(C3=CC=CC=C3)=N2
 -> 
C1=CC=CC=C1NC2=NC=CC(C3=CC=C(F)C=C3)=N2


C1=CN(n2cncn2)C(Nc2ccc(C3=CCCN3)cc2)C1


'''









