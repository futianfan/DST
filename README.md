# DST (differentiable scaffolding tree)





















## 1. setup

please see `conda.yml`

```bash
conda activate tdc
source activate differentiable_molecular_graph
```
































## 2. data preparation


### Raw Data 

We use `ZINC` database, which contains around 250K drug-like molecules. 
input is `raw_data/zinc.tab`, each row is a SMILES. 
`raw_data/zinc.tab_head50` is an example file that has first 50 rows. 



### Oracle

Molecular property is evaluated by oracle. 

* `JNK3`
* `GSK3B` 
* `LogP` 
* `QED` 
* `SA` normalized SA to (0,1): see `sa.py`



### Task 

* `jnkgsk`
* `qedsajnkgsk`
* `qed`
* `logp`
* `jnk`
* `gsk`



### Labeling

We use oracle to evaluate molecule's properties to obtain the labels for molecules. 

- input
  - `raw_data/zinc.tab`: all the smiles in ZINC, around 250K. 

- output
  - `data/zinc_*.txt`: `*` can be QED, LogP, JNK3, GSK3B, etcs. 

```bash  
python src/data_zinc.py 
```

### Generate Vocabulary 
In this project, the basic unit is substructure, which contains frequent atoms and rings. The vocabulary is the set of all these atoms and rings. 

- substructure
  - basic unit in molecule tree, including rings and atoms. 

- input
  - `raw_data/zinc.tab`: all the smiles in ZINC, around 250K. 

- output
  - `data/all_vocabulary.txt`: including all the substructures in ZINC.   
  - `data/selected_vocabulary.txt`: vocabulary, frequent substructures. 


```bash 
python src/data_generate_vocabulary.py
```

### 2.6 data cleaning  

We want to remove the molecules that contains substructure that is not in vocabulary 


- input 
  - `data/selected_vocabulary.txt`: vocabulary 
  - `raw_data/zinc.tab`: all the smiles in ZINC
  - `data/zinc_QED.txt` 


- output
  - `data/zinc_QED_clean.txt`


```bash 
python src/data_cleaning.py 
```


































## 3. pipeline 

### 3.1 train GNN

- input 
  - `data/zinc_QED_clean.txt`: **training data** includes `(X,y)` pairs, where `X` is the molecule, `y` is the label. `y = GNN(X)`

- output 
  - `save_model/model_epoch_*.ckpt`: saved GNN model. 

- log
  - `"valid_loss_folder/" + prop + ".pkl"` save the valid loss. 

```bash 
python src/train_{$prop}.py 
```
`prop` is `qed`, `logp`, `jnk`, `gsk`, `jnkgsk`, ... 
for logp, GNN minimizes MSE, for other tasks, it minimizes binary cross entropy. 



### 3.2 de novo generation 

```bash
python src/denovo_{$prop}.py 
```

- input 
  - `save_model/model_epoch_*.ckpt`: saved GNN model. 

- output 
  - `result/denovo_{$prop}.pkl`: generated moles. 



### 3.3 evaluate 

```bash

```
- input 
  - `result/denovo_{$prop}.pkl`: generated moles. 

- output 





## code interpretation 


### build DST 



### optimize DST 



### sampling from DST 



### DPP 




### utility 














