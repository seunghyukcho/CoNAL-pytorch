import json
import torch
import argparse
import importlib
from tqdm import tqdm
from pathlib import Path
from munch import munchify
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from arguments import add_test_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_test_args(parser)
    args = parser.parse_args()

    print('Loading configurations...')
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_file = ckpt_dir / 'best_model.pth'
    ckpt = torch.load(ckpt_file)

    config_file = ckpt_dir / 'args.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
        config = munchify(config)

    task_name = config.task
    task_module = importlib.import_module(f'tasks.{task_name}')
    task_dataset = getattr(task_module, 'Dataset')

    transform = None
    if task_name == 'labelme':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    test_dataset = task_dataset(config, args.test_data, transforms=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)
    classifier = getattr(task_module, 'Classifier')(config)
    classifier.load_state_dict(ckpt['classifier'])
    classifier = classifier.to(args.device)

    with torch.no_grad():
        classifier.eval()
        correct = 0
        for x, y in tqdm(test_loader):
            x, y = x.to(args.device), y.to(args.device)
            pred = classifier(x)
            pred = torch.argmax(pred, dim=1)
            correct += torch.sum(torch.eq(pred, y)).item()

        print(f'Test Accuracy: {correct / len(test_dataset)}')
