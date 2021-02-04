import argparse

tasks = ['labelme', 'music']


def get_task_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--task', type=str, choices=tasks)
    return parser


def add_model_args(parser):
    group = parser.add_argument_group('model')
    group.add_argument('--input_dim', type=int,
                       help="Input dimension of CoNAL.")
    group.add_argument('--n_class', type=int,
                       help="Number of classes for classification.")
    group.add_argument('--n_annotator', type=int,
                       help="Number of annotators that labeled the data.")
    group.add_argument('--emb_dim', type=int, default=20,
                       help="Dimension of embedding in auxiliary network of CoNAL (Default: 20).")


def add_train_args(parser):
    group = parser.add_argument_group('train')
    group.add_argument('--seed', type=int, default=7777,
                       help="Random seed (Default: 7777).")
    group.add_argument('--epochs', type=int, default=10,
                       help="Number of epochs for training (Default: 10).")
    group.add_argument('--batch_size', type=int, default=32,
                       help="Number of instances in a batch (Default: 32).")
    group.add_argument('--lr', type=float, default=1e-5,
                       help="Learning rate (Default: 1e-5).")
    group.add_argument('--log_interval', type=int, default=1,
                       help="Log interval (Default: 1).")
    group.add_argument('--task', type=str, choices=tasks,
                       help="Task name for training.")
    group.add_argument('--train_data', type=str,
                       help="Root directory of train data.")
    group.add_argument('--valid_data', type=str,
                       help="Root directory of validation data.")
    group.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda',
                       help="Device going to use for training (Default: cuda).")
    group.add_argument('--save_dir', type=str, default='checkpoints/',
                       help="Folder going to save model checkpoints (Default: checkpoints/).")
    group.add_argument('--log_dir', type=str, default='logs/',
                       help="Folder going to save logs (Default: logs/).")
    group.add_argument('--scale', type=float, default=0,
                       help="Scale of regularization term (Default: 0).")


def add_test_args(parser):
    group = parser.add_argument_group('test')
    group.add_argument('--batch_size', type=int, default=32,
                       help="Number of instances in a batch (Default: 32).")
    group.add_argument('--test_data', type=str,
                       help="Root directory of test data.")
    group.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu',
                       help="Device going to use for training (Default: cpu).")
    group.add_argument('--ckpt_dir', type=str,
                       help="Directory which contains the checkpoint and args.json.")
