# COMP 5331 2022/23 Fall Group 9

[GitHub Page](https://github.com/RYY0722/TENT_COMP5331)

Project type: Implementation

Reference: [Task-Adaptive Few-shot Node Classification (arxiv.org)](https://arxiv.org/pdf/2206.11972.pdf)

Environment requirements:

Python >=3.9

Install pyg and pytorch packages with the following commands

```sh
## pytorch==1.10.1+cpu
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -c pytorch -y
## pyg
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.10.1+cpu.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.10.1+cpu.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.10.1+cpu.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.10.1+cpu.html
pip install -U -i https://pypi.tuna.tsinghua.edu.cn/simple torch_geometric

```



Other requirements

```
matplotlib
pathlib
```



## Available benchmark datasets

- [Cora-full](https://arxiv.org/abs/1707.03815)
- [CoAuthorCS](https://kddcup2016.azurewebsites.net/)
- [AmazonClothing]([Graph Prototypical Networks for Few-shot Learning on Attributed Networks | Proceedings of the 29th ACM International Conference on Information & Knowledge Management](https://dl.acm.org/doi/10.1145/3340531.3411922))
- [AmazonElectronics](https://arxiv.org/pdf/1506.08839.pdf)
- [DBLP]([AMiner](https://www.aminer.org/citation))
- [Email-EU]([Local Higher-Order Graph Clustering | Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining](https://dl.acm.org/doi/10.1145/3097983.3098069))
- [Reddit]([GraphSAGE (stanford.edu)](http://snap.stanford.edu/graphsage/))
- [obgn-arxiv]([Node Property Prediction | Open Graph Benchmark (stanford.edu)](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv))
- [WikiCSDataset]([pmernyei/wiki-cs-dataset: Wiki-CS: A Wikipedia-Based Benchmark for Graph Neural Networks (github.com)](https://github.com/pmernyei/wiki-cs-dataset))

The preprocessed datasets can be found at [dataset](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yruanaf_connect_ust_hk/EqliD2zZ4X9CiUPbovRfSn8Ba47Bd1tLtiCtjBCCFZzxXg?e=BwoHsv)

## Available benchmark backbones and pipelines

### Backbones

- GCN [[1609.02907\] Semi-Supervised Classification with Graph Convolutional Networks (arxiv.org)](https://arxiv.org/abs/1609.02907)
- GAT [[1710.10903\] Graph Attention Networks (arxiv.org)](https://arxiv.org/abs/1710.10903)
- GraphSage [[1706.02216\] Inductive Representation Learning on Large Graphs (arxiv.org)](https://arxiv.org/abs/1706.02216)

### Pipelines

- Vanilla [[1711.04043\] Few-Shot Learning with Graph Neural Networks (arxiv.org)](https://arxiv.org/abs/1711.04043)
- PN [Prototypical Networks for Few-shot Learning (neurips.cc)](https://proceedings.neurips.cc/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Paper.pdf)
- GPN [Graph Prototypical Networks for Few-shot Learning on Attributed Networks (arxiv.org)](https://arxiv.org/pdf/2006.12739.pdf)
- MAML [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (mlr.press)](https://proceedings.mlr.press/v70/finn17a/finn17a.pdf)
- TENT [Task-Adaptive Few-shot Node Classification | Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining](https://dl.acm.org/doi/abs/10.1145/3534678.3539265)

|         | GCN  |   GAT   | GraphSage |
| :-----: | :--: | :-----: | :-------: |
| Vanilla |  √   |    √    |     √     |
|  MAML   |      | Default |           |
|   PN    |  √   |    √    |     √     |
|   GPN   |  √   |    √    |     √     |
|  TENT   |      | Default |           |

\* TENT and MAML is only implemented with its default backbones. 

## Usage

```sh
sh train.sh
```

