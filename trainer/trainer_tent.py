from pathlib import Path
import torch.optim as optim
import torch
from utils.utils import *
from torch.nn import functional as F
from importlib import import_module
from trainer.trainer_basic import Trainer as Basic
from model.loss import *
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import torch.nn as nn
from model.backbones import GCN_dense, GCN_emb
from torch.nn import Linear

class Trainer(Basic):
    def __init__(self, args):
        self.in_feat = args.in_feat
        self.nhid = args.hidden
        self.nclass = args.nclass
        self.args = args
        self.save_dir = Path(self.args.save_dir)

        self.model = GCN_dense(nfeat=self.nhid,
                                    nhid=args.hidden,
                                    nclass=self.nclass,
                                    dropout=args.dropout)

        self.GCN_model = GCN_emb(nfeat=self.in_feat,
                                nhid=args.hidden,
                                nclass=self.nclass,
                                dropout=args.dropout)

        self.classifier = Linear(self.nhid, self.nclass)

        self.optimizer = optim.Adam([{'params': self.model.parameters()}, {'params': self.classifier.parameters()},{'params': self.GCN_model.parameters()}],
                                        lr=args.lr, weight_decay=args.weight_decay)
        self.loss_f = nn.CrossEntropyLoss()
        self.Q = args.Q


    def train(self, features, adj, labels, class_selected, id_support, id_query, n_way, k_shot, **kargs):
        class_train_dict = kargs['class_train_dict']
        class_test_dict = kargs['class_test_dict']
        class_valid_dict = kargs['class_valid_dict']
        idx_train = kargs['idx_train']
        self.model.train()
        self.optimizer.zero_grad()
        results = self.loop( features, adj, labels, 'train', n_way, k_shot, class_train_dict, class_test_dict, class_valid_dict, idx_train)
        acc = results['acc']
        loss = results['loss']
        loss.backward()
        self.optimizer.step()
        return acc, loss

    def test(self, features, adj, labels, class_selected, id_support, id_query, n_way, k_shot, **kargs):
        class_train_dict = kargs['class_train_dict']
        class_test_dict = kargs['class_test_dict']
        class_valid_dict = kargs['class_valid_dict']
        idx_train = kargs['idx_train']
        mode = kargs['mode']
        self.model.eval()
        if mode == 'test':
            results = self.loop(features, adj, labels, 'test', n_way, k_shot, class_train_dict, class_test_dict, class_valid_dict, idx_train)
        elif mode == 'val':
            results = self.loop(features, adj, labels, 'valid', n_way, k_shot, class_train_dict, class_test_dict, class_valid_dict, idx_train)

        acc = results['acc']
        loss = results['loss']
        return acc, loss
    

    def loop(self, features, adj, labels, mode, n_way, k_shot, class_train_dict, class_test_dict, class_valid_dict, idx_train):

        if mode == 'train':
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()
        # adj_dense = adj.to_dense()
        emb_features= self.GCN_model(features, adj)

        target_idx = []
        target_graph_adj_and_feat = []
        support_graph_adj_and_feat = []

        pos_node_idx = []

        if mode == 'train':
            class_dict = class_train_dict
        elif mode == 'test':
            class_dict = class_test_dict
        elif mode=='valid':
            class_dict = class_valid_dict
        
        classes = np.random.choice(list(class_dict.keys()), n_way, replace=False).tolist()


        pos_graph_adj_and_feat=[]   
        for i in classes:
            # sample from one specific class
            sampled_idx=np.random.choice(class_dict[i], k_shot+self.Q, replace=False).tolist()
            pos_node_idx.extend(sampled_idx[:k_shot])
            target_idx.extend(sampled_idx[k_shot:])

            class_pos_idx=sampled_idx[:k_shot]

            if k_shot==1 and torch.nonzero(adj[class_pos_idx,:]).shape[0]==1:
                pos_class_graph_adj=adj[class_pos_idx,class_pos_idx].reshape([1,1])
                pos_graph_feat=emb_features[class_pos_idx]
            else:
                pos_graph_neighbors = torch.nonzero(adj[class_pos_idx, :].sum(0)).squeeze()

                pos_graph_adj = adj[pos_graph_neighbors, :][:, pos_graph_neighbors]

                pos_class_graph_adj=torch.eye(pos_graph_neighbors.shape[0]+1,dtype=torch.float)

                pos_class_graph_adj[1:,1:]=pos_graph_adj

                pos_graph_feat=torch.cat([emb_features[class_pos_idx].mean(0,keepdim=True),emb_features[pos_graph_neighbors]],0)

            pos_graph_adj_and_feat.append((pos_class_graph_adj, pos_graph_feat))

        target_graph_adj_and_feat=[]  
        for node in target_idx:
            if torch.nonzero(adj[node,:]).shape[0]==1:
                pos_graph_adj=adj[node,node].reshape([1,1])
                pos_graph_feat=emb_features[node].unsqueeze(0)
            else:
                pos_graph_neighbors = torch.nonzero(adj[node, :]).squeeze()
                pos_graph_neighbors = torch.nonzero(adj[pos_graph_neighbors, :].sum(0)).squeeze()
                pos_graph_adj = adj[pos_graph_neighbors, :][:, pos_graph_neighbors]
                pos_graph_feat = emb_features[pos_graph_neighbors]

            target_graph_adj_and_feat.append((pos_graph_adj, pos_graph_feat))

        class_generate_emb=torch.stack([sub[1][0] for sub in pos_graph_adj_and_feat],0).mean(0)

        parameters=self.model.generater(class_generate_emb)

        gc1_parameters=parameters[:(self.nhid+1)*self.nhid*2]
        gc2_parameters=parameters[(self.nhid+1)*self.nhid*2:]

        gc1_w=gc1_parameters[:self.nhid*self.nhid*2].reshape([2,self.nhid,self.nhid])
        gc1_b=gc1_parameters[self.nhid*self.nhid*2:].reshape([2,self.nhid])

        gc2_w=gc2_parameters[:self.nhid*self.nhid*2].reshape([2,self.nhid,self.nhid])
        gc2_b=gc2_parameters[self.nhid*self.nhid*2:].reshape([2,self.nhid])

        self.model.eval()
        ori_emb = []
        for i, one in enumerate(target_graph_adj_and_feat):
            sub_adj, sub_feat = one[0], one[1]
            ori_emb.append(self.model(sub_feat, sub_adj, gc1_w, gc1_b, gc2_w, gc2_b).mean(0))  # .mean(0))

        target_embs = torch.stack(ori_emb, 0)

        class_ego_embs=[]
        for sub_adj, sub_feat in pos_graph_adj_and_feat:
            class_ego_embs.append(self.model(sub_feat,sub_adj,gc1_w,gc1_b,gc2_w,gc2_b)[0])
        class_ego_embs=torch.stack(class_ego_embs,0)

        target_embs=target_embs.reshape([n_way,self.Q,-1]).transpose(0,1)
        
        support_features = emb_features[pos_node_idx].reshape([n_way,k_shot,-1])
        class_features=support_features.mean(1)
        taus=[]
        for j in range(n_way):
            taus.append(torch.linalg.norm(support_features[j]-class_features[j],-1).sum(0))
        taus=torch.stack(taus,0)

        similarities=[]
        for j in range(self.Q):
            class_contras_loss, similarity=InforNCE_Loss(target_embs[j], self.args.dataset,class_ego_embs/taus.unsqueeze(-1),tau=0.5)
            similarities.append(similarity)

        loss_supervised=self.loss_f(self.classifier(emb_features[idx_train]), labels[idx_train])

        loss=loss_supervised

        labels_train=labels[target_idx]
        for j, class_idx in enumerate(classes[:n_way]):
            labels_train[labels_train==class_idx]=j
            
        loss+=self.loss_f(torch.stack(similarities,0).transpose(0,1).reshape([n_way*self.Q,-1]), labels_train)
    
        acc_train = accuracy(torch.stack(similarities,0).transpose(0,1).reshape([n_way*self.Q,-1]), labels_train)

        if mode=='valid' or mode=='test':
            support_features = l2_normalize(emb_features[pos_node_idx].detach().cpu()).numpy()
            query_features = l2_normalize(emb_features[target_idx].detach().cpu()).numpy()

            support_labels=torch.zeros(n_way*k_shot,dtype=torch.long)
            for i in range(n_way):
                support_labels[i * k_shot:(i + 1) * k_shot] = i

            query_labels=torch.zeros(n_way*self.Q,dtype=torch.long)
            for i in range(n_way):
                query_labels[i * self.Q:(i + 1) * self.Q] = i

            clf = LogisticRegression(penalty='l2',
                                    random_state=0,
                                    C=1.0,
                                    solver='lbfgs',
                                    max_iter=1000,
                                    multi_class='multinomial')
            clf.fit(support_features, support_labels.numpy())
            query_ys_pred = clf.predict(query_features)

            acc_train = metrics.accuracy_score(query_labels, query_ys_pred)

        return {'acc': acc_train.item(), 'loss': loss}