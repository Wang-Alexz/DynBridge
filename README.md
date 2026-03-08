## DynBridge: Bridging Imagination and Control through Interaction Dynamics for Robot Manipulation


## Installation

```
conda env create -f conda_env.yml
```

## Dataset Preprocessing

We first need to download the raw LIBERO datasets and then preprocess them with [Cotracker](https://arxiv.org/abs/2307.07635):

```
python data_generation/generate_libero_traj.py
```

## Training

To train DynB on LIBERO, use the following command.

```
python -u train.py suite/task=libero_object suite.num_train_steps=201000 suite.flow_weight=0.01 alpha=0.5 suite.track_ts=15
```

## Evaluation

The evaluation can be executed by this command.
```
sh eval_find.sh
```

## Other LIBERO-set Commands for Training

```
python -u train.py suite/task=libero_goal suite.num_train_steps=201000 suite.flow_weight=1e-4 alpha=0.5 suite.track_ts=15 

python -u train.py suite/task=libero_spatial suite.num_train_steps=201000 suite.flow_weight=0.01 alpha=0.5 suite.track_ts=15 

python -u train.py suite/task=libero_10 suite.num_train_steps=401000 suite.flow_weight=1e-4 alpha=0.9 suite.track_ts=10 

python -u train.py suite/task=libero_90 suite.num_train_steps=401000 suite.flow_weight=0.01 alpha=0.1 suite.track_ts=10
```
