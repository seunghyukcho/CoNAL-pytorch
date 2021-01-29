import json

import torch
import argparse
import importlib
import torch.nn as nn
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from model import CoNAL
from arguments import add_train_args, add_model_args


if __name__ == "__main__":
    task_parser = argparse.ArgumentParser(add_help=False)
    task_parser.add_argument('--task', type=str, choices=['labelme', 'music'],
                             help="Task name for training.")

    task_name = task_parser.parse_known_args()[0].task
    task_module = importlib.import_module(f'tasks.{task_name}')
    task_dataset = getattr(task_module, 'Dataset')

    parser = argparse.ArgumentParser()
    add_train_args(parser)
    add_model_args(parser)
    getattr(task_module, 'add_task_args')(parser)
    args = parser.parse_args()

    transform = None
    if task_name == 'labelme':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    print('Loading train dataset...')
    train_dataset = task_dataset(args, args.train_data, is_train=True, transforms=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    print('Loading validation dataset...')
    valid_dataset = task_dataset(args, args.valid_data, transforms=transform)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size)

    print('Building model...')
    classifier = getattr(task_module, 'Classifier')(args)
    model = CoNAL(
        args.input_dim,
        args.n_class,
        args.n_annotator,
        classifier,
        annotator_dim=args.n_annotator,
        embedding_dim=args.emb_dim
    )
    if args.device == 'cuda':
        model = nn.DataParallel(model)
    model = model.to(args.device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
    optimizer = Adam(model.parameters(), lr=args.lr)

    print('Start training!')
    best_accuracy = 0
    writer = SummaryWriter(args.log_dir)
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_correct = 0
        model.train()
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
            f'Epoch: {epoch + 1} | Training | '
            f'Total Loss: {total_loss / len(train_dataset) } | '
            f'Total Accuracy of Classifier: {total_correct / len(train_dataset)}'
        )
        writer.add_scalar('train_loss', total_loss / len(train_dataset), epoch)
        writer.add_scalar('train_accuracy', total_correct / len(train_dataset), epoch)

        total_correct = 0
        model.eval()
        for x, y in valid_loader:
            x, y = x.to(args.device), y.to(args.device)
            pred = model.classifier(x)
            pred = torch.argmax(pred, dim=1)
            total_correct += torch.sum(torch.eq(pred, y)).item()
        print(
            f'Epoch: {epoch + 1} | Validation | '
            f'Total Accuracy of Classifier: {total_correct / len(valid_dataset)}'
        )
        writer.add_scalar('valid_accuracy', total_correct / len(valid_dataset), epoch)

        if best_accuracy < total_correct:
            best_accuracy = total_correct
            checkpoint_dir = Path(args.save_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'auxiliary_network': model.auxiliary_network.state_dict(),
                'noise_adaptation_layer': model.noise_adaptation_layer.state_dict(),
                'classifier': model.classifier.state_dict()
            }, checkpoint_dir / 'best_model.pth')

            with open(checkpoint_dir / 'args.json', 'w') as f:
                json.dump(args.__dict__, f, indent=2)
