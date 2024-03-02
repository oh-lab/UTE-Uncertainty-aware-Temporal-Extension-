# Uncertainty-aware Temporal Extension
This repository contains the code for the AAAI'24 paper [Learning Uncertainty-Aware Temporally-Extended Actions](https://arxiv.org/abs/2402.05439).


## Setup & Requirements
This code was developed with python 3.6.13 and torch 1.8.1.

```conda create -n ute python=3.6```

```conda actviate ute```

```conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge```

To install requirements:
```setup
pip install -r requirements.txt
```

To train the model(s) in the paper, run this command:

## Chain MDP experiments

```cd chain_mdp```

### $\epsilon$z-greedy from [Temporally Extended $\epsilon$-Greedy Exploration](https://arxiv.org/abs/2006.01782)

```python main_DQN.py --agent ez_greedy --cuda 0 --input-dim 50 --max-episodes=1000```

### TempoRL

```python main_UTE.py --agent tdqn --cuda 0 --input-dim 50 --max-episodes=1000 --skip-net-max-skips=10```

### UTE

```python main_UTE.py --agent ute --cuda 0 --input-dim 50 --max-episodes=1000 --skip-net-max-skips=10 --uncertainty-factor=2.0```


## Gridworlds experiments

```cd grid_atari```

### DDQN

```python gridworlds.py --agent q --env lava```

### TempoRL

```python gridworlds.py --agent sq --env lava --max-skips 7```

### UTE

```python gridworlds.py --agent ute --env lava --max-skips 7 --uncertainty-factor -1.5```


## Atari experiments

```cd grid_atari```

### DDQN

```python atari.py --env qbert --env-max-steps 10000 --agent dqn --out-dir ./ --episodes 20000 --training-steps 2500000 --eval-after-n-steps 10000 --seed 12345 --84x84 --eval-n-episodes 3```

### TempoRL

``` python atari.py --env qbert --env-max-steps 10000 --agent tdqn --out-dir ./ --episodes 20000 --training-steps 2500000 --eval-after-n-steps 10000 --seed 12345 --84x84 --eval-n-episodes 3```

### UTE

```python atari.py --env qbert --env-max-steps 10000 --agent ute --out-dir ./ --episodes 20000 --training-steps 2500000 --eval-after-n-steps 10000 --seed 12345 --84x84 --eval-n-episodes 3 --uncertainty-factor -0.5```

### UTE with adaptive lambda

```python atari.py --env qbert --env-max-steps 10000 --agent ute_bandit --out-dir ./ --episodes 20000 --training-steps 2500000 --eval-after-n-steps 10000 --seed 12345 --84x84 --eval-n-episodes 3 --uncertainty-factor -0.5```


The Chain MDP environment is based on the code for the paper:
[Randomized Value functions via Multiplicative Normalizing Flows.
Ahmed Touati, Harsh Satija, Joshua Romoff, Joelle Pineau, Pascal Vincent. UAI 2019](https://arxiv.org/abs/1806.02315)
```@article{touati2018randomized,
  title={Randomized value functions via multiplicative normalizing flows},
  author={Touati, Ahmed and Satija, Harsh and Romoff, Joshua and Pineau, Joelle and Vincent, Pascal},
  journal={arXiv preprint arXiv:1806.02315},
  year={2018}
}
```
The gridworlds/Atari environmnet and TempoRL agent is based on the code for the paper:
[TempoRL: Learning When to Act](https://arxiv.org/abs/2106.05262)
```@inproceedings{biedenkapp-icml21,
  author    = {André Biedenkapp and Raghu Rajan and Frank Hutter and Marius Lindauer},
  title     = {{T}empo{RL}: Learning When to Act},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning (ICML 2021)},
  year = {2021},
  month     = jul,
}
```