import rdkit 
from rdkit import Chem
from rdkit.Chem import Draw

import sys 

smiles = sys.argv[1]



mol = Chem.MolFromSmiles(smiles) 
Draw.MolToImageFile(mol, "figure/tmp.png", size = (300,180))



