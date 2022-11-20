from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
from model.backbones import GCN
from utils.utils import *
from utils.args import get_args
from torch.nn import functional as F
from collections import defaultdict
from importlib import import_module
import json
from utils.dataset import shot_way_info, load_data
from pathlib import Path
import networkx as nx


if __name__ == '__main__':
    # Setup
    args = get_args()
    # args.pipeline = 'GPN'
    # args.model = "GAT"
    args.cuda = args.use_cuda and torch.cuda.is_available()
    num_repeat = args.num_repeat
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        print("Using GPU acceleration... ")
    m = import_module('trainer.trainer_'+args.pipeline.lower())
    _pipeline = getattr(m, 'Trainer')
    # Loading dataset
    dataset = args.dataset
    adj, features, labels, idx_train, idx_valid, idx_test, n1s, n2s, class_train_dict, class_test_dict, class_valid_dict, id_by_class, degrees = load_data(dataset)
    class_list_valid = list(class_valid_dict)
    class_list_test = list(class_test_dict)
    class_list_train = list(class_train_dict)
    shot_way_pairs = shot_way_info[dataset]['pairs']
    if args.model in ['GAT', 'GraghSage']:
        D = nx.DiGraph(adj.to_dense().numpy())
        edge_lst = nx.to_pandas_edgelist(D)
        edge_lst = [edge_lst['source'], edge_lst['target']]
        adj = torch.Tensor(edge_lst).long()
        del D, edge_lst
    if args.pipeline.lower() == 'tent':
        adj = adj.to_dense()
    # create dict for storing results
    results=defaultdict(dict)

    # Main for loops
    for n_way, k_shot in shot_way_pairs:
        meta_test_acc_total = np.zeros((num_repeat))
        meta_test_f1_total = np.zeros((num_repeat))
        n_query = shot_way_info[dataset]['Q']
        args.Q = n_query
        meta_test_num = 50
        meta_valid_num = 50
        print("Training %s for on %s (%d-way %d-shot) Q: %d" % (args.model, dataset, n_way, k_shot, n_query))
        for repeat in range(num_repeat):
            args.nclass = labels.max().item() + 1
            args.in_feat = features.shape[1]
            t = _pipeline(args)
            if args.cuda:
                t.model.cuda()
                features = features.cuda()
                adj = adj.cuda()
                labels = labels.cuda()
                degrees = degrees.cuda()
            print("Repeat %d: Training %s for on %s (%d-way %d-shot)" % (repeat, args.model, dataset, n_way, k_shot))
            # Sampling a pool of tasks for validation/testing
            valid_pool = [task_generator(id_by_class, class_list_valid, n_way, k_shot, n_query) for i in range(meta_valid_num)]
            test_pool = [task_generator(id_by_class, class_list_test, n_way, k_shot, n_query) for i in range(meta_test_num)]

            # Train model
            t_total = time.time()
            best_valid_acc = 0
            meta_train_acc = []
            for episode in range(args.epochs):
                id_support, id_query, class_selected = \
                    task_generator(id_by_class, class_list_train, n_way, k_shot, n_query)

                acc_train, loss_train = t.train(features, adj, labels, class_selected, id_support, id_query, n_way, k_shot, \
                                                degrees=degrees, id_by_class=id_by_class, class_list_train=class_list_train, n_query=n_query, \
                                                idx_train=idx_train, class_train_dict=class_train_dict, class_valid_dict=class_valid_dict, class_test_dict=class_test_dict)
                meta_train_acc.append(acc_train)
                # Test every "test_every" epochs OR Test at the last epoch
                if (episode > 0 and episode % args.test_every == 0) or (episode == args.epochs-1):    
                    print("-------Episode {}-------".format(episode))
                    print("Meta-Train_Accuracy: {}".format(np.array(meta_train_acc).mean(axis=0)))
                    meta_val_acc = []
                    meta_test_acc = []
                    # validation
                    for idx in range(meta_valid_num):
                        id_support, id_query, class_selected = valid_pool[idx]
                        acc_val, loss_val = t.test(features, adj, labels, class_selected, id_support, id_query, n_way, k_shot, \
                                                degrees=degrees, id_by_class=id_by_class, class_list_train=class_list_train, n_query=n_query, \
                                                idx_train=idx_train, class_train_dict=class_train_dict, class_valid_dict=class_valid_dict, class_test_dict=class_test_dict, mode='val')
                        meta_val_acc.append(acc_val)
                    print("Meta-valid_Accuracy: {}".format(np.array(meta_val_acc).mean(axis=0)))
                    # testing
                    for idx in range(meta_test_num):
                        id_support, id_query, class_selected = test_pool[idx]
                        acc_test, loss_test = t.test(features, adj, labels, class_selected, id_support, id_query, n_way, k_shot, \
                                                degrees=degrees, id_by_class=id_by_class, class_list_train=class_list_train, n_query=n_query, \
                                                idx_train=idx_train, class_train_dict=class_train_dict, class_valid_dict=class_valid_dict, class_test_dict=class_test_dict, mode='test')
                        meta_test_acc.append(acc_test)
                    # update best test accuracy if validation accuracy improves
                    valid_acc = np.array(meta_test_acc).mean(axis=0)
                    if valid_acc > best_valid_acc:
                        best_valid_acc = valid_acc
                        meta_test_acc_total[repeat] = np.array(meta_test_acc).mean(axis=0)
                    print("Meta-Test_Accuracy: {}".format(meta_test_acc_total[repeat]))

            print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

            ## save results at the end of every repeat
            for i in range(repeat+1):
                print(meta_test_acc_total[i])
                results['{}-way {}-shot {}-repeat'.format(n_way,k_shot,i)]= meta_test_acc_total[i]
                json.dump(results,open(save_dir / '{}_result_{}.json'.format(args.pipeline + '_'+args.model, dataset),'w'), indent=4) 
            accs=[]
            for i in range(num_repeat):
                accs.append(results['{}-way {}-shot {}-repeat'.format(n_way,k_shot,i)])
        ## save results for the dataset
        results['{}-way {}-shot'.format(n_way,k_shot)]=np.mean(accs)
        results['{}-way {}-shot_print'.format(n_way,k_shot)]='acc: {:.4f}'.format(np.mean(accs))
        json.dump(results,open(save_dir / '{}_result_{}.json'.format(args.pipeline + '_'+args.model, dataset),'w'), indent=4) 

    print("Done :)")