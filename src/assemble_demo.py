from chemutils import * 

s = "C1=CC=CC=C1C2CCCCN2"
ring = 'C1CCCCC1'
bondtype_list = [rdkit.Chem.rdchem.BondType.SINGLE, rdkit.Chem.rdchem.BondType.DOUBLE]

origin_mol = Chem.rdchem.RWMol(Chem.MolFromSmiles(s))

# def add_fragment_at_position(editmol, position_idx, fragment, new_bond):
whole_set = set()
for i in range(12):
	for bondtype in bondtype_list:
		new_smiles_set = add_fragment_at_position(origin_mol, position_idx= i, fragment = ring, new_bond = bondtype)
		whole_set = whole_set.union(new_smiles_set)

whole_set = list(whole_set)
for i,smiles in enumerate(whole_set):
	draw_smiles(smiles, "figure/"+str(i)+".png")
print(whole_set, len(whole_set))

draw_smiles(s, 'figure/assemble_0.png')

'''
ring-atom 

C1=CC=CC=C1C2=CC=C(Cl)C=N2
C1=CC=CC=C1C2=CC(Cl)=CC=N2
C1=CC=CC=C1C2=C(Cl)C=CC=N2

'''