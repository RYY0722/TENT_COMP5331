from pathlib import Path
from unittest import result
import torch.optim as optim
import torch
from utils.utils import *
from torch.nn import functional as F
from importlib import import_module

class Trainer():
    def __init__(self, args):
        self.args = args
        self.save_dir = Path(self.args.save_dir)
        m = import_module('model.backbones')
        _model = getattr(m, args.model)
        self.model =  _model(self.args.in_feat, self.args.hidden, self.args.dropout)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def loop(self, features, adj, labels, class_selected, id_support, id_query, n_way, k_shot):
        output = self.model(features, adj)['emb'][id_query]
        output = F.log_softmax(output, dim=1)
        labels_new = torch.LongTensor([class_selected.index(i) for i in labels[id_query]])
        if self.args.cuda:
            labels_new = labels_new.cuda()
        _loss = F.nll_loss(output, labels_new)

        if self.args.cuda:
            output = output.cpu().detach()
            labels_new = labels_new.cpu().detach()
        _acc = accuracy(output, labels_new)
        return{'acc': _acc, 'loss':_loss}
        
    def train(self, features, adj, labels, class_selected, id_support, id_query, n_way, k_shot, **kwargs):
        self.model.train()
        self.optimizer.zero_grad()
        results = self.loop(features, adj, labels, class_selected, id_support, id_query, n_way, k_shot)
        acc = results['acc']
        loss = results['loss']
        loss.backward()
        self.optimizer.step()
        return acc, loss

    def test(self, features, adj, labels, class_selected, id_support, id_query, n_way, k_shot, **kwargs):
        self.model.eval()
        results = self.loop( features, adj, labels, class_selected, id_support, id_query, n_way, k_shot)
        acc = results['acc']
        loss = results['loss']
        return acc, loss
