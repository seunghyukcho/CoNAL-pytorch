import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils import data
from torchvision import transforms


def add_dataset_args(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument('--degree', type=int, default=0,
                       help='Degrees of random affine augmentation.')
    group.add_argument('--shear', type=int, default=15,
                       help='Shearing of random affine augmentation.')


class Dataset(data.Dataset):
    def __init__(self, args, root_dir, is_train=False):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / 'images'
        self.is_train = is_train
        self.degree = args.degree
        self.shear = args.shear

        self.transforms = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        if self.is_train:
            augmentations = transforms.RandomChoice([
                transforms.RandomAffine(degrees=self.degree, shear=self.shear),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomResizedCrop(224)
            ])
            self.transforms.insert(0, augmentations)
        self.transforms = transforms.Compose(self.transforms)

        self.labels = np.loadtxt(self.root_dir / 'labels.txt', dtype=np.float32)
        with open(self.root_dir / 'filenames.txt', 'r') as f:
            self.image_paths = f.read().splitlines()
        if self.is_train:
            self.annotations = np.loadtxt(self.root_dir / 'annotations.txt', dtype=np.int64)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        x = Image.open(self.image_dir / self.image_paths[idx])
        x = self.transforms(x)
        y = self.labels[idx]

        if self.is_train:
            annotation = self.annotations[idx]
            return x, y, annotation
        else:
            return x, y
