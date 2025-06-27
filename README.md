# Mining Users' Preferences in Recommender Systems: A Reinforced Knowledge Graph Reasoning Approach
The implementation of the proposed DualPGPR model.
## Datasets
The datasets used in the experiments are from Amazon available at http://jmcauley.ucsd.edu/data/amazon/. The preprocessed datasets are in the `data` folder.

## Requirements
The required packages are listed in the `requirements.txt` file.

## Usage
1. Preprocess the datasets: 
```python
python preprocess.py --dataset <dataset_name>
```

2. Train the TransE embeddings:
```python
python train_transe_model.py --dataset <dataset_name>
```
Arguments:
- `--dataset`: the name of the dataset (one of beauty, cloth, cell)
-  `--name` : the name of the model
-  `--epochs` : the number of epochs
-  `--batch_size` : the batch size
-  `--lr` : the learning rate
-  `--weight_decay` : the weight decay
-  `--l2_lambda` : the lambda for l2 regularization
-  `--max_grad_norm` : the maximum gradient norm
-  `--embed_dim` : the embedding dimension
-  `--num_neg_samples` : the number of negative samples
-  `--steps_per_checkpoint` : the steps per checkpoint
-  
3. Train RL agent:
```python
python train_agent.py
```
Arguments:
- `--dataset` : the name of the dataset (one of beauty, cloth, cell)
-  `--max_acts` : the maximum number of actions
-  `--max_path_len` : the maximum number of hops
-  `--gamma` : the discount factor
-  `--lr` : the learning rate
-  `--batch_size` : the batch size
-  `--epochs` : the number of epochs
-  `--ent_weight` : the weight for calculating the entropy loss
-  `--act_dropout` : the action dropout rate
-  `--state_history` : the number of historical states prior to the current state
-  `--hidden` : the hidden size of the FC layer
1. Test the model:
```python
python test_agent.py
```
Arguments:
- `--dataset` : the name of the dataset (one of beauty, cloth, cell)
- `--max_acts` : the maximum number of actions
- `--max_path_len` : the maximum number of hops
- `--run_path` : whether to generate reasoning paths or not
- `--run_eval` : whether to run evaluation or not
- `--get_neg_paths` : whether to get negative paths or not, turn off if testing evaluation metrics only
  

## References
[1] Yongfeng Zhang, Qingyao Ai, Xu Chen, W. Bruce Croft. "Joint Representation Learning for Top-N Recommendation with Heterogeneous Information Sources". In Proceedings of CIKM. 2017.

[2] Yikun Xian, Zuohui Fu, S. Muthukrishnan, Gerard de Melo, Yongfeng Zhang. "Reinforcement Knowledge Graph Reasoning for Explainable Recommendation." In Proceedings of SIGIR. 2019.

[3] Wang, X., Li, Q., Yu, D., Li, Q., & Xu, G. (2024). Reinforced path reasoning for counterfactual explainable recommendation. IEEE Transactions on Knowledge and Data Engineering.