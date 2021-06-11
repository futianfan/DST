import numpy as np 
from tdc import Oracle
qed = Oracle(name = 'qed')
# logp = Oracle(name = 'logp')
jnk = Oracle('jnk3')
gsk = Oracle('gsk3b')
from sa import sa 
def oracle(smiles):
	scores = [qed(smiles), sa(smiles), jnk(smiles), gsk(smiles)]
	return np.mean(scores)
	# return qed_logp_fusion(*scores)

# s = 
smiles_lst = ['Nc1ccc(-c2ccnc(Nc3ccc(-n4cncn4)cc3)n2)cc1', "Nc1ccc(-c2ccnc(Nc3ccc(-n4cnc(Cl)n4)cc3)n2)cc1", 'Clc1ncn(-c2ccc(Nc3nccc(-c4ccccc4)n3)cc2)n1']

for s in smiles_lst:
	print(oracle(s))

