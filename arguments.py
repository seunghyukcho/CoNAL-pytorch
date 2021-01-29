def add_train_args(parser):
    group = parser.add_argument_group('train')
    group.add_argument('--epochs', type=int, default=10,
                       help="Number of epochs for training.")
    group.add_argument('--batch_size', type=int, default=32,
                       help="Number of instances in a batch.")
    group.add_argument('--lr', type=float, default=1e-5,
                       help="Learning rate.")
    group.add_argument('--log_interval', type=int, default=10,
                       help="Log interval.")
    group.add_argument('--task', type=str, choices=['labelme', 'music'],
                       help="Task name for training.")
    group.add_argument('--train_data', type=str,
                       help="Root directory of train data.")
    group.add_argument('--valid_data', type=str,
                       help="Root directory of validation data.")
    group.add_argument('--device', type=str, choices=['cpu', 'cuda'],
                       help="Device going to use for training.")
    group.add_argument('--save_dir', type=str, default='checkpoints/',
                       help="Folder going to save model checkpoints.")
    group.add_argument('--log_dir', type=str, default='logs/',
                       help="Folder going to save logs.")


def add_model_args(parser):
    group = parser.add_argument_group('model')
    group.add_argument('--input_dim', type=int,
                       help="Input dimension of CoNAL.")
    group.add_argument('--n_class', type=int,
                       help="Number of classes for classification.")
    group.add_argument('--n_annotator', type=int,
                       help="Number of annotators that labeled the data.")
    group.add_argument('--emb_dim', type=int, default=20,
                       help="Dimension of embedding in auxiliary network of CoNAL.")
