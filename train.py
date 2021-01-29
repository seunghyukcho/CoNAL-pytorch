import torch
import argparse
import importlib
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from model import CoNAL


def add_args(parser):
    train_args = parser.add_argument_group('train')
    train_args.add_argument('--epochs', type=int, default=10,
                            help="Number of epochs for training.")
    train_args.add_argument('--batch_size', type=int, default=32,
                            help="Number of instances in a batch.")
    train_args.add_argument('--lr', type=float, default=1e-5,
                            help="Learning rate.")
    train_args.add_argument('--task', type=str, choices=['labelme', 'music'],
                            help="Task name for training.")
    train_args.add_argument('--train_data', type=str,
                            help="Root directory of train data.")
    train_args.add_argument('--valid_data', type=str,
                            help="Root directory of validation data.")
    train_args.add_argument('--device', type=str, choices=['cpu', 'cuda'],
                            help="Device going to use for training.")

    model_args = parser.add_argument_group('model')
    model_args.add_argument('--input_dim', type=int,
                            help="Input dimension of CoNAL.")
    model_args.add_argument('--n_class', type=int,
                            help="Number of classes for classification.")
    model_args.add_argument('--n_annotator', type=int,
                            help="Number of annotators that labeled the data.")
    model_args.add_argument('--emb_dim', type=int, default=20,
                            help="Dimension of embedding in auxiliary network of CoNAL.")


if __name__ == "__main__":
    task_parser = argparse.ArgumentParser(add_help=False)
    parser = argparse.ArgumentParser()
    task_parser.add_argument('--task', type=str, choices=['labelme', 'music'],
                             help="Task name for training.")

    task_name = task_parser.parse_known_args()[0].task
    task_module = importlib.import_module(f'tasks.{task_name}')
    task_args = getattr(task_module, 'add_task_args')
    task_dataset = getattr(task_module, 'Dataset')

    add_args(parser)
    task_args(parser)
    args = parser.parse_args()

    train_dataset = task_dataset(args, args.train_data, is_train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = task_dataset(args, args.valid_data)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size)

    classifier = getattr(task_module, 'Classifier')(args)
    model = CoNAL(
        args.input_dim,
        args.n_class,
        args.n_annotator,
        classifier,
        annotator_dim=args.n_class,
        embedding_dim=args.emb_dim
    )
    model = model.to(args.device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
    classifier_criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_correct = 0
        for x, y, annotation in train_loader:
            model.zero_grad()
            x, y, annotation = x.to(args.device), y.to(args.device), annotation.to(args.device)
            annotator = torch.eye(args.n_annotator).to(args.device)
            ann_out, cls_out = model(x, annotator)

            ann_out = torch.reshape(ann_out, (-1, 8))
            annotation = annotation.view(-1)

            loss = criterion(ann_out, annotation)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            pred = torch.argmax(cls_out, dim=1)
            total_correct += torch.sum(torch.eq(pred, y)).item()

        print(
            f'Epoch: {epoch} | '
            f'Total Loss: {total_loss / len(train_dataset) } | '
            f'Total Accuracy of Classifier: {total_correct / len(train_dataset)}'
        )
