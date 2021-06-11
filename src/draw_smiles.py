import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Draw


def load_vocabulary():
	datafile = "data/selected_vocabulary.txt"
	with open(datafile, 'r') as fin:
		lines = fin.readlines()
	vocabulary = [line.split()[0] for line in lines]
	return vocabulary 

vocabulary = load_vocabulary()
bondtype_list = [rdkit.Chem.rdchem.BondType.SINGLE, rdkit.Chem.rdchem.BondType.DOUBLE]


def ith_substructure_is_atom(i):
    substructure = vocabulary[i]
    return True if len(substructure)==1 else False

def word2idx(word):
    return vocabulary.index(word)


def smiles2fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, useChirality=False)
    return np.array(fp)
    ### shape: (1024,)


## similarity of two SMILES 
def similarity(a, b):
    if a is None or b is None: 
        return 0.0
    amol = Chem.MolFromSmiles(a)
    bmol = Chem.MolFromSmiles(b)
    if amol is None or bmol is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(bmol, 2, nBits=2048, useChirality=False)
    return DataStructs.TanimotoSimilarity(fp1, fp2) 


def similarity_matrix(smiles_lst):
    n = len(smiles_lst)
    sim_matrix = np.eye(n)
    mol_lst = [Chem.MolFromSmiles(smiles) for smiles in smiles_lst]
    fingerprint_lst = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=False) for mol in mol_lst]
    for i in range(n):
        fp1 = fingerprint_lst[i]
        for j in range(i+1,n):
            fp2 = fingerprint_lst[j]
            sim = DataStructs.TanimotoSimilarity(fp1, fp2)
            sim_matrix[i,j] = sim_matrix[j,i] = sim
    return sim_matrix 


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True) ### todo double check
    else:
        return None


def smiles2mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    Chem.Kekulize(mol)
    return mol 

## input: smiles, output: word lst;  
def smiles2word(smiles):
    mol = smiles2mol(smiles)
    if mol is None:
        return None 
    word_lst = []

    cliques = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques_smiles = []
    for clique in cliques:
        clique_smiles = Chem.MolFragmentToSmiles(mol, clique, kekuleSmiles=True)
        cliques_smiles.append(clique_smiles)
    atom_not_in_rings_list = [atom.GetSymbol() for atom in mol.GetAtoms() if not atom.IsInRing()]
    return cliques_smiles + atom_not_in_rings_list 

## is_valid_smiles 
def is_valid(smiles):
    word_lst = smiles2word(smiles)
    word_set = set(word_lst)
    return word_set.issubset(vocabulary)     


def is_valid_mol(mol):
    try:
        smiles = Chem.MolToSmiles(mol)
    except:
        return False 
    if smiles.strip() == '':
        return False 
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return False 
    return True 

def substr_num(smiles):
    mol = smiles2mol(smiles)
    clique_lst = [list(x) for x in Chem.GetSymmSSSR(mol)]
    return len(clique_lst)


def draw_smiles(smiles, figfile_name):
    mol = Chem.MolFromSmiles(smiles)
    Draw.MolToImageFile(mol, figfile_name, size = (300,180))
    return 




def draw_vocabulary(vocabulary):
    for idx, smiles in enumerate(vocabulary):
        draw_smiles(smiles, "figure/vocab_" + str(idx) + ".png")

from tdc import Oracle
qed = Oracle(name = 'qed')
sa = Oracle(name = 'sa')
jnk = Oracle(name = 'JNK3')
gsk = Oracle(name = 'GSK3B')

if __name__ == "__main__":
    # draw_vocabulary(vocabulary)
    import sys 
    smiles = sys.argv[1]
    draw_smiles(smiles, 'figure/tmp.png')
    print(qed(smiles), sa(smiles), jnk(smiles), gsk(smiles)) 

'''

C=C1COCCN1N1CCC=C1Nc1nccc(-c2cnccn2)n1      0.9021781153084742 -1.0281787967266767 0.46 0.72 


'''





















