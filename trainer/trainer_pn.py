from pathlib import Path
import torch.optim as optim
import torch
from utils.utils import *
from torch.nn import functional as F
from importlib import import_module
from trainer.trainer_basic import Trainer as Basic
class Trainer(Basic):
    def __init__(self, args):
        self.args = args
        self.save_dir = Path(self.args.save_dir)
        m = import_module('model.backbones')
        _model = getattr(m, args.model)
        self.model =  _model(args.in_feat, self.args.hidden, self.args.dropout)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def loop(self, features, adj, labels, class_selected, id_support, id_query, n_way, k_shot):

        result = self.model(features, adj)
        embeddings, scores = result['emb'], result['score']
        z_dim = embeddings.size()[1]

        # embedding lookup
        support_embeddings = embeddings[id_support]
        support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
        query_embeddings = embeddings[id_query]
        n_query = len(id_query)
        
        # construct prototype
        prototypes = support_embeddings.sum(1)
        dists = euclidean_dist(query_embeddings, prototypes)
        p = k_shot * n_way
        data_shot, data_query = support_embeddings, query_embeddings

        proto = data_shot
        proto = proto.reshape(k_shot, n_way, -1).mean(dim=0)

        label = torch.LongTensor([class_selected.index(i) for i in labels[id_query]])

        logits = euclidean_dist(data_query, proto)
        loss = F.cross_entropy(logits, label)
        acc = count_acc(logits, label)
        return {'loss':loss, 'acc':acc}

