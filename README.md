# DST: Differentiable Scaffolding Tree for Molecule Optimization 

This repository hosts [DST (Differentiable Scaffolding Tree for Molecule Optimization)](https://openreview.net/forum?id=w_drCosT76&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2022%2FConference%2FAuthors%23your-submissions)) (Tianfan Fu*, Wenhao Gao*, Cao Xiao, Jacob Yasonik, Connor W. Coley, Jimeng Sun), which enables a gradient-based optimization on a chemical graph. 


## Table Of Contents

- Installation 
- Data and Setup
  - raw data 
  - oracle
  - optimization task 
  - labelling
  - generate vocabulary 
  - data cleaning  
  - limit oracle setting 
- Learning and Inference
  - train graph neural network (GNN)
  - de novo molecule design 
  - evaluate  
- Contact 



## 1. Installation 

To install locally, we recommend to install from `pip` and `conda`. Please see `conda.yml` for the package dependency. 
 
```bash
conda create -n dst python=3.7 
pip install torch 
pip install PyTDC 
conda install -c rdkit rdkit 
```


The following command activate conda environment. 
```bash
conda activate dst
conda activate differentiable_molecular_graph
```



## 2. Data and Setup


### Raw Data 

We use [`ZINC`](https://tdcommons.ai/generation_tasks/molgen/) database, which contains around 250K drug-like molecules. 
Due to space limitation, we upload a subset of ZINC in `raw_data/zinc.tab_head50`, which contains xxx molecules, each row is a SMILES. 



### Oracle

Oracle is a property evaluator and is a function whose input is molecular structure, and output is the ground truth value of the property. 
We consider following oracles: 
* `JNK3`: biological activity to JNK3, ranging from 0 to 1.
* `GSK3B` biological activity to GSK3B, ranging from 0 to 1. 
* `QED`: Quantitative Estimate of Drug-likeness, ranging from 0 to 1. 
* `SA`: Synthetic Accessibility, we normalize SA to (0,1). 
* `LogP`: solubility and synthetic accessibility of a compound. It ranges from negative infinity to positive infinity. 

For all the property scores above, higher is more desirable. 


### Optimization Task 

There are two kinds of optimization tasks: single-objective and multi-objective optimization. 
Multi-objective optimization contains `jnkgsk` (JNK3 + GSK3B), `qedsajnkgsk` (QED + SA + JNK3 + GSK3B). 


### Labelling

We use oracle to evaluate molecule's properties to obtain the labels for training graph neural network. 
```bash  
python src/data_zinc.py 
```
- input
  - `raw_data/zinc.tab`: all the smiles in ZINC, around 250K. 
- output
  - `data/zinc_*.txt`: `*` can be QED, LogP, JNK3, GSK3B, etcs. 



### Generate Vocabulary 
In this project, the basic unit is `substructure`, which can be atoms and single rings. 
The vocabulary is the set of frequent `substructures`. 
```bash 
python src/data_generate_vocabulary.py
```
- input
  - `raw_data/zinc.tab`: all the smiles in ZINC, around 250K. 
- output
  - `data/all_vocabulary.txt`: including all the substructures in ZINC.   
  - `data/selected_vocabulary.txt`: vocabulary, frequent substructures. 

### 2.6 data cleaning  

We remove the molecules that contains substructure that is not in vocabulary. 

```bash 
python src/data_cleaning.py 
```

- input 
  - `data/selected_vocabulary.txt`: vocabulary 
  - `raw_data/zinc.tab`: all the smiles in ZINC
  - `data/zinc_QED.txt`:  
- output
  - `data/zinc_QED_clean.txt`


### 2.7 limit oracle setting 

```
head -10000 data/zinc_QED_clean.txt > data/zinc_QED_clean_10K.txt
```





## 3. Learning and Inference 

In our setup, we restrict the number of oracle calls in both training GNN and de novo design. 

### 3.1 train graph neural network (GNN)

It corresponds to Section 3.2 in the paper. 
```bash 
python src/train.py $prop $train_oracle
```
- `prop` represent the property to optimize, including `qed`, `logp`, `jnk`, `gsk`, `jnkgsk`, `qedsajnkgsk`.  
- `train_oracle` is number of oracle calls in training GNN. 
- input 
  - `data/zinc_QED_clean.txt`: **training data** includes `(SMILES,y)` pairs, where `SMILES` is the molecule, `y` is the label. `y = GNN(SMILES)`
- output 
  - `save_model/model_epoch_*.ckpt`: saved GNN model. 
- log
  - `"loss/{$prop}.pkl"` save the valid loss. 

For example, 
```bash 
python src/train.py jnkgsk 5000 
```

### 3.2 de novo molecule design 

It corresponds to Section 3.3 and 3.4 in the paper.  

```bash
python src/denovo.py $prop $denovo_oracle
```
- `prop` represent the property to optimize, including `qed`, `logp`, `jnk`, `gsk`, `jnkgsk`, `qedsajnkgsk`. 
- `denovo_oracle` is number of oracle calls. 
- input 
  - `save_model/{$prop}_*.ckpt`: saved GNN model. * is number of iteration or epochs. 
- output 
  - `result/{$prop}.pkl`: set of generated molecules. 

For example, 
```bash 
python src/denovo.py jnkgsk 5000 
```

### 3.3 evaluate 

```bash
python src/evaluate.py $prop  
```
- input 
  - `result/{$prop}.pkl`
- output 
  - `diversity`, `novelty`, `average property` of top-100 molecules with highest property. 

For example, 
```bash 
python src/evaluate.py jnkgsk 
```

<!-- ## Example  -->




## Contact 
Please contact futianfan@gmail.com or gaowh19@gmail.com for help or submit an issue. 


## Cite Us
If you found this package useful, please cite [our paper](https://openreview.net/forum?id=w_drCosT76&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2022%2FConference%2FAuthors%23your-submissions)):
```
@article{fu2020differentiable,
  title={Differentiable Scaffolding Tree for Molecule Optimization},
  author={Tianfan Fu*, Wenhao Gao*, Cao Xiao, Jacob Yasonik, Connor W. Coley, Jimeng Sun},
  journal={International Conference on Learning Representation (ICLR)},
  year={2022}
}
```






