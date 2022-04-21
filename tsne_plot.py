from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import sys
import argparse
import torch
import numpy as np
import yaml
import json
import random
from random import shuffle
from collections import OrderedDict
import dataloaders
from dataloaders.utils import *
from torch.utils.data import DataLoader
import learners

def plot(args, seed):
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    # prepare dataloader
    Dataset = None
    if args.dataset == 'CIFAR10':
        Dataset = dataloaders.iCIFAR10
        num_classes = 10
    elif args.dataset == 'CIFAR100':
        Dataset = dataloaders.iCIFAR100
        num_classes = 100
    elif args.dataset == 'TinyIMNET':
        Dataset = dataloaders.iTinyIMNET
        num_classes = 200
    else:
        Dataset = dataloaders.H5Dataset
        num_classes = 100

    # load tasks
    class_order = np.arange(num_classes).tolist()
    class_order_logits = np.arange(num_classes).tolist()

    # seed = 1

    if seed > 0 and args.rand_split:
        
        print('=============================================')
        print('Shuffling....')
        print('pre-shuffle:' + str(class_order))
        random.seed(seed)
        random.shuffle(class_order)
        print('post-shuffle:' + str(class_order))
        print('=============================================')
    
    tasks = []
    tasks_logits = []
    p = 0
    while p < num_classes:
        inc = args.other_split_size if p > 0 else args.first_split_size
        tasks.append(class_order[p:p+inc])
        tasks_logits.append(class_order_logits[p:p+inc])
        p += inc
    num_tasks = len(tasks)
    task_names = [str(i+1) for i in range(num_tasks)]

    # number of transforms per image
    k = 1
    if args.fm_loss: 
        k = 2
    ky = 1

    # datasets and dataloaders
    train_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=args.train_aug)
    train_transformb = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=args.train_aug, hard_aug=True)
    test_transform  = dataloaders.utils.get_transform(dataset=args.dataset, phase='test', aug=args.train_aug)
    
    print(args.dataroot, args.dataset, args.labeled_samples, args.unlabeled_task_samples, args.l_dist, args.ul_dist,
                            tasks, seed, args.rand_split, args.validation, args.repeat)

    train_dataset = Dataset(args.dataroot, args.dataset, args.labeled_samples, args.unlabeled_task_samples, train=True, lab = True,
                            download=True, transform=TransformK(train_transform, train_transform, ky), l_dist=args.l_dist, ul_dist=args.ul_dist,
                            tasks=tasks, seed=seed, rand_split=args.rand_split, validation=args.validation, kfolds=args.repeat)

    train_dataset_ul = Dataset(args.dataroot, args.dataset, args.labeled_samples, args.unlabeled_task_samples, train=True, lab = False,
                            download=True, transform=TransformK(train_transform, train_transformb, k), l_dist=args.l_dist, ul_dist=args.ul_dist,
                            tasks=tasks, seed=seed, rand_split=args.rand_split, validation=args.validation, kfolds=args.repeat)
                            
    test_dataset  = Dataset(args.dataroot, args.dataset, train=False,
                            download=False, transform=test_transform, l_dist=args.l_dist, ul_dist=args.ul_dist,
                            tasks=tasks, seed=seed, rand_split=args.rand_split, validation=args.validation, kfolds=args.repeat)

    # in case tasks reset...
    tasks = train_dataset.tasks

    # Prepare the Learner (model)
    learner_config = {'num_classes': num_classes,
                      'lr': args.lr,
                      'ul_batch_size': args.ul_batch_size,
                      'tpr': args.tpr,
                      'oodtpr': args.oodtpr,
                      'momentum': args.momentum,
                      'weight_decay': args.weight_decay,
                      'schedule': args.schedule,
                      'schedule_type': args.schedule_type,
                      'model_type': args.model_type,
                      'model_name': args.model_name,
                      'ood_model_name': args.ood_model_name,
                      'out_dim': args.force_out_dim,
                      'optimizer': args.optimizer,
                      'gpuid': args.gpuid,
                      'pl_flag': args.pl_flag,
                      'fm_loss': args.fm_loss,
                      'weight_aux': args.weight_aux,
                      'memory': args.memory,
                      'distill_loss': args.distill_loss,
                      'co': args.co,
                      'FT': args.FT,
                      'DW': args.DW,
                      'num_labeled_samples': args.labeled_samples,
                      'num_unlabeled_samples': args.unlabeled_task_samples,
                      'super_flag': args.l_dist == "super",
                      'no_unlabeled_data': args.no_unlabeled_data,
                      'dynamic_threshold': args.dynamic_threshold,
                      'fm_thresh': args.fm_thresh,
                      'fm_epsilon': args.fm_epsilon,
                      'threshold_warmup': args.threshold_warmup,
                      'non_linear_mapping': args.non_linear_mapping
                      }
    learner = learners.__dict__[args.learner_type].__dict__[args.learner_name](learner_config)
    print(learner.model)
    print('#parameter of model:', learner.count_parameter())

    acc_table = OrderedDict()
    acc_table_pt = OrderedDict()
    if args.learner_name == 'DistillMatch' and len(task_names) > 1 and not args.oracle_flag:
        run_ood = {}
    else:
        run_ood = None
    save_table_ssl = []
    save_table = []
    save_table_pc = -1*np.ones((num_tasks,num_tasks))
    pl_table = [[],[],[],[]]
    temp_dir = args.log_dir + '/temp'
    if not os.path.exists(temp_dir): 
        os.makedirs(temp_dir)

    # for oracle
    out_dim_add = 0

    ###
    # Training
    ###
    # Feed data to learner and evaluate learner's performance
    if args.max_task > 0:
        max_task = min(args.max_task, len(task_names))
    else:
        max_task = len(task_names)

    for i in range(max_task):

        # set seeds
        random.seed(seed*100 + i)
        np.random.seed(seed*100 + i)
        torch.manual_seed(seed*100 + i)
        torch.cuda.manual_seed(seed*100 + i)

        train_name = task_names[i]
        print('======================', train_name, '=======================')

        # load dataset for task
        task = tasks_logits[i]
        print("\nTask:", task, len(task))
        prev = sorted(set([k for task in tasks_logits[:i] for k in task]))

        # if oracle
        if args.oracle_flag:
            train_dataset.load_dataset(prev, i, train=False)
            train_dataset_ul.load_dataset(prev, i, train=False)
            learner = learners.__dict__[args.learner_type].__dict__[args.learner_name](learner_config)
            out_dim_add += len(task)
        else:
            train_dataset.load_dataset(prev, i, train=True)
            train_dataset_ul.load_dataset(prev, i, train=True)
            out_dim_add = len(task)

        # load dataset with memory
        train_dataset.append_coreset(only=False)

        # load dataloader
        train_loader_l = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=int(args.workers / 2))
        train_loader_ul = DataLoader(train_dataset_ul, batch_size=args.ul_batch_size, shuffle=True, drop_last=False, num_workers=int(args.workers / 2))
        train_loader_ul_task = DataLoader(train_dataset_ul, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=int(args.workers / 2))
        train_loader = dataloaders.SSLDataLoader(train_loader_l, train_loader_ul)

        # add valid class to classifier
        learner.add_valid_output_dim(out_dim_add)
        print("Out dim add:", out_dim_add)
        # Learn
        test_dataset.load_dataset(prev, i, train=False)
        test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)
        model_save_dir = args.log_dir + '/models/repeat-'+str(seed+1)+'/task-'+task_names[i]+'/'
        if not os.path.exists(model_save_dir): 
            os.makedirs(model_save_dir)
        learner.plot_tsne(train_loader, train_dataset, train_dataset_ul, model_save_dir, test_loader, task_num = int(task_names[i]), visualisation_dir = args.visualisation_dir)

def create_args():
    
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()

    # Standard Args
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                         help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--log_dir', type=str, default="outputs/out",
                         help="Save experiments results in dir for future plotting!")
    parser.add_argument('--model_type', type=str, default='mlp', help="The type (mlp|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--model_name', type=str, default='MLP', help="The name of actual model for the backbone")
    parser.add_argument('--force_out_dim', type=int, default=2, help="Set 0 to let the task decide the required output dimension")
    parser.add_argument('--learner_type', type=str, default='default', help="The type (filename) of learner")
    parser.add_argument('--learner_name', type=str, default='NormalNN', help="The class name of learner")
    parser.add_argument('--optimizer', type=str, default='SGD', help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--dataroot', type=str, default='data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--dataset', type=str, default='CIFAR100', help="CIFAR10|CIFAR100|TinyIMNET")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--schedule', nargs="+", type=int, default=[2],
                        help="The list of epoch numbers to reduce learning rate by factor of 0.1. Last number is the end epoch")
    parser.add_argument('--schedule_type', type=str, default='cosine',
                        help="decay, cosine")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--ul_batch_size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8, help="#Thread for dataloader")
    parser.add_argument('--validation', default=False, action='store_true', help='Evaluate on fold of training dataset rather than testing data')
    parser.add_argument('--FT', default=False, action='store_true', help='finetune distillation')
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")

    # OOD Args
    parser.add_argument('--ood_model_name', type=str, default=None, help="The name of actual model for the backbone ood")
    parser.add_argument('--tpr', type=float, default=0.95, help="tpr for ood calibration of class network")
    parser.add_argument('--oodtpr', type=float, default=0.95, help="tpr for ood calibration of ood network")

    # SSL Args
    parser.add_argument('--weight_aux', type=float, default=1.0, help="Auxillery weight, usually used for trading off unsupervised and supervised losses")
    parser.add_argument('--labeled_samples', type=int, default=50000, help='Number of labeled samples in ssl')
    parser.add_argument('--unlabeled_task_samples', type=int, default=0, help='Number of unlabeled samples in each task in ssl')
    parser.add_argument('--fm_loss', default=False, action='store_true', help='Use fix-match loss with classifier (WARNING: currently only pseudolabel)')
    parser.add_argument('--pl_flag', default=False, action='store_true', help='use pseudo-labeled ul data for DM')
    
    # GD Args
    parser.add_argument('--no_unlabeled_data', default=False, action='store_true')
    parser.add_argument('--distill_loss', nargs="+", type=str, default='C', help='P, C, Q')
    parser.add_argument('--co', type=float, default=1., metavar='R',
                    help='out-of-distribution confidence loss ratio (default: 0.)')

    # CL Args
    parser.add_argument('--first_split_size', type=int, default=2, help="size of first CL task")
    parser.add_argument('--other_split_size', type=int, default=2, help="size of remaining CL tasks")              
    parser.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',
                        help="Allow data augmentation during training")
    parser.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
                        help="Randomize the classes in splits")
    parser.add_argument('--l_dist', type=str, default='vanilla', help="vanilla|super")
    parser.add_argument('--ul_dist', type=str, default=None, help="none|vanilla|super - if none, copy l dist")
    parser.add_argument('--oracle_flag', default=False, action='store_true', help='Upper bound for oracle')
    parser.add_argument('--max_task', type=int, default=-1, help="number tasks to perform; if -1, then all tasks")
    parser.add_argument('--memory', type=int, default=0, help="size of memory for replay")
    parser.add_argument('--DW', default=False, action='store_true', help='dataset balancing')

    parser.add_argument('--dynamic_threshold', default = False, help='set dynamic threshold for consistency regularisation')
    parser.add_argument('--fm_thresh', default = 0.85, help='set dynamic threshold for consistency regularisation')
    parser.add_argument('--fm_epsilon', default = 0.000001, type = float)
    parser.add_argument('--threshold_warmup', default = False, type = bool)
    parser.add_argument('--non_linear_mapping', default = False, type = bool)
    parser.add_argument('--visualisation_dir', default = "./visualisation", type = bool)
    
    return parser

def get_args(argv):
    parser=create_args()
    args = parser.parse_args(argv)
    return args

# want to save everything printed to outfile
class Logger(object):
    def __init__(self, name):
        self.terminal = sys.stdout
        self.log = open(name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()

if __name__ == '__main__':
    args = get_args(sys.argv[1:])

    # determinstic backend
    torch.backends.cudnn.deterministic=True

    # duplicate output stream to output file
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    log_out = args.log_dir + '/output.log'
    sys.stdout = Logger(log_out)

    # save args
    json.dump(
            vars(args), open(args.log_dir + '/args.yaml', "w")
        )

    avg_final_acc = np.zeros(args.repeat)
    avg_ood = {'tpr': {}, 'fpr': {}, 'de': {}, 'roc-auc': {}}
    
    start_r = 0
    for r in range(start_r, args.repeat):

        print('************************************')
        print('* STARTING TRIAL ' + str(r+1))
        print('************************************')

        # set random seeds
        seed = r
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        ####### T-sne plotting
        plot(args, seed)