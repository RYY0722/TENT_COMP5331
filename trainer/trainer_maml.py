from pathlib import Path
import torch.optim as optim
import torch
from utils.utils import *
from model.backbones import GCN
from torch.nn import functional as F
from importlib import import_module
from trainer.trainer_basic import Trainer as Basic
from collections import OrderedDict
class Trainer(Basic):
    def __init__(self, args):
        self.args = args
        self.save_dir = Path(self.args.save_dir)
        
        self.model =  GCN(args.in_feat, self.args.hidden, self.args.dropout)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
    
    def train(self, features, adj, labels, class_selected, id_support, id_query, n_way, k_shot, **kargs):
        id_by_class = kargs['id_by_class']
        class_list_train = kargs['class_list_train']
        n_query = kargs['n_query']
        loss_q = 0
        acc_q = 0
        # from main import id_by_class, class_list_train, n_query
        for i in range(self.args.task_num):
            output = self.model(features, adj)['emb'][id_support]
            labels_new = torch.LongTensor([class_selected.index(i) for i in labels[id_support]])
            if self.args.cuda:
                labels_new = labels_new.cuda()
            loss_train = F.nll_loss(output, labels_new)

            fast_weights = OrderedDict((name, param) for (name, param) in self.model.named_parameters() if name.find('fc') == -1)
            grad = torch.autograd.grad(loss_train, self.model.parameters(), create_graph=True, allow_unused=True)
            fast_weights = OrderedDict((name, param - self.args.lr * grad) for ((name, param), grad) in zip(fast_weights.items(), grad))

            id_support_new, id_querry_new, class_selected_new = task_generator(id_by_class, class_list_train, n_way, k_shot, n_query)
            output = self.model.forward_pesudo(features, adj, fast_weights)[id_querry_new]
            labels_new = torch.LongTensor([class_selected_new.index(i) for i in labels[id_querry_new]])
                
            if self.args.cuda:
                labels_new = labels_new.cuda()
            loss_q += F.nll_loss(output, labels_new)       
            acc_q += accuracy(output, labels_new)
     
        loss_q = loss_q / self.args.task_num
        acc_q = acc_q / self.args.task_num
        self.optimizer.zero_grad()
        loss_q.backward()
        self.optimizer.step()
        return acc_q, loss_q

    
