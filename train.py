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
    # Read task argument first, and determine the other arguments
    task_parser = argparse.ArgumentParser(add_help=False)
    task_parser.add_argument('--task', type=str, choices=['labelme', 'music'])

    task_name = task_parser.parse_known_args()[0].task
    task_module = importlib.import_module(f'tasks.{task_name}')
    task_dataset = getattr(task_module, 'Dataset')

    parser = argparse.ArgumentParser()
    add_train_args(parser)
    add_model_args(parser)
    getattr(task_module, 'add_task_args')(parser)
    args = parser.parse_args()

    # Seed settings
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Transform settings based on task
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
    model = model.to(args.device)

    # Ignore annotators labeling which is -1
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
    optimizer = Adam(model.parameters(), lr=args.lr)

    print('Start training!')
    best_accuracy = 0
    writer = SummaryWriter(args.log_dir)
    for epoch in range(args.epochs):
        train_loss = 0.0
        train_correct = 0
        model.train()
        for x, y, annotation in train_loader:
            model.zero_grad()

            # Annotator embedding matrix (in this case, just a identity matrix)
            annotator = torch.eye(args.n_annotator)

            # Move the parameters to device given by argument
            x, y, annotation, annotator = x.to(args.device), y.to(args.device), annotation.to(args.device), annotator.to(args.device)
            ann_out, cls_out = model(x, annotator)

            # Calculate loss of annotators' labeling
            ann_out = torch.reshape(ann_out, (-1, args.n_class))
            annotation = annotation.view(-1)
            loss = criterion(ann_out, annotation)

            # Regularization term
            confusion_matrices = model.noise_adaptation_layer
            matrices = confusion_matrices.local_confusion_matrices - confusion_matrices.global_confusion_matrix
            for matrix in matrices:
                loss -= args.scale * torch.linalg.norm(matrix)

            # Update model weight using gradient descent
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Calculate classifier accuracy
            pred = torch.argmax(cls_out, dim=1)
            train_correct += torch.sum(torch.eq(pred, y)).item()
        print(
            f'Epoch: {epoch + 1} |  Training  | '
            f'Total Accuracy of Classifier: {train_correct / len(train_dataset)} |'
            f'Total Loss: {train_loss}'
        )

        # Validation
        with torch.no_grad():
            valid_correct = 0
            model.eval()
            for x, y in valid_loader:
                x, y = x.to(args.device), y.to(args.device)
                pred = model(x)
                pred = torch.argmax(pred, dim=1)
                valid_correct += torch.sum(torch.eq(pred, y)).item()
        print(
            f'Epoch: {epoch + 1} | Validation | '
            f'Total Accuracy of Classifier: {valid_correct / len(valid_dataset)}'
        )

        # Save tensorboard log
        if epoch % args.log_interval == 0:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_accuracy', train_correct / len(train_dataset), epoch)
            writer.add_scalar('valid_accuracy', valid_correct / len(valid_dataset), epoch)

        # Save the model with highest accuracy on validation set
        if best_accuracy < valid_correct:
            best_accuracy = valid_correct
            checkpoint_dir = Path(args.save_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'auxiliary_network': model.auxiliary_network.state_dict(),
                'noise_adaptation_layer': model.noise_adaptation_layer.state_dict(),
                'classifier': model.classifier.state_dict()
            }, checkpoint_dir / 'best_model.pth')

            with open(checkpoint_dir / 'args.json', 'w') as f:
                json.dump(args.__dict__, f, indent=2)

