from pathlib import Path
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

    def loop(self, features, adj, labels, class_selected, id_support, id_query, n_way, k_shot, degrees):

        result = self.model(features, adj)
        embeddings, scores = result['emb'], result['score']
        z_dim = embeddings.size()[1]

        # embedding lookup
        support_embeddings = embeddings[id_support]
        support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
        query_embeddings = embeddings[id_query]

        # node importance
        support_degrees = torch.log(degrees[id_support].view([n_way, k_shot]))
        support_scores = scores[id_support].view([n_way, k_shot])
        support_scores = torch.sigmoid(support_degrees * support_scores).unsqueeze(-1)
        support_scores = support_scores / torch.sum(support_scores, dim=1, keepdim=True)
        support_embeddings = support_embeddings * support_scores

        # compute loss
        prototype_embeddings = support_embeddings.sum(1)
        dists = euclidean_dist(query_embeddings, prototype_embeddings)
        output = F.log_softmax(-dists, dim=1)

        labels_new = torch.LongTensor([class_selected.index(i) for i in labels[id_query]])
        if self.args.cuda:
            labels_new = labels_new.cuda()
        loss_train = F.nll_loss(output, labels_new)

        if self.args.cuda:
            output = output.cpu().detach()
            labels_new = labels_new.cpu().detach()
        acc_train = accuracy(output, labels_new)

        return {'loss':loss_train, 'acc':acc_train}
        
    def train(self, features, adj, labels, class_selected, id_support, id_query, n_way, k_shot, **kwargs):
        degrees = kwargs['degrees']
        self.model.train()
        self.optimizer.zero_grad()
        results = self.loop(features, adj, labels, class_selected, id_support, id_query, n_way, k_shot, degrees)
        acc = results['acc']
        loss = results['loss']
        loss.backward()
        self.optimizer.step()
        return acc, loss

    def test(self, features, adj, labels, class_selected, id_support, id_query, n_way, k_shot, **kwargs):
        degrees = kwargs['degrees']
        self.model.eval()
        results = self.loop( features, adj, labels, class_selected, id_support, id_query, n_way, k_shot, degrees)
        acc = results['acc']
        loss = results['loss']
        return acc, loss
