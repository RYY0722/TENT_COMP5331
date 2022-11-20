import argparse
def get_args():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train.')
    parser.add_argument('--test_every', type=int, default=20,
                        help='Test per x epochs')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')

    parser.add_argument('--dataset', type=str, help='Dataset name', default='email')
    parser.add_argument('--save_dir', type=str, help='Directory for saving the results', default='results/')

    #### added by ryy
    parser.add_argument('--model', type=str, help='Model name: GCN/GAT/GPN/GraghSage', default='GCN')
    parser.add_argument('--pipeline', type=str, help='Pipeline name: Basic/PN/MAML', default='TENT')
    parser.add_argument('--num_repeat', type=int, help='Repeat times', default=2)
    #### TENT specific
    parser.add_argument('--hidden2', type=int, help='Dimension of the second hidden layer', default=16)
    #### MAML specific
    parser.add_argument('--task_num', type=int, help='Task number', default=32)
   
    '''
    Deprecated; but might be useful
    ''' 
    # parser.add_argument('--way', type=int, default=5, help='way.')
    # parser.add_argument('--shot', type=int, default=5, help='shot.')
    # parser.add_argument('--qry', type=int, help='k shot for query set', default=10)
    return parser.parse_args()
