import numpy as np
from pathlib import Path
from torch.utils import data


def add_dataset_args(parser):
    group = parser.add_argument_group('dataset')


class Dataset(data.Dataset):
    def __init__(self, args, root_dir, is_train=False, transforms=None):
        self.root_dir = Path(root_dir)
        self.is_train = is_train
        self.transforms = transforms

        self.music = np.loadtxt(self.root_dir / 'data.csv', delimiter=',', dtype=np.float32)
        self.labels = np.loadtxt(self.root_dir / 'labels.csv', delimiter=',', dtype=np.int64)
        if self.is_train:
            self.annotations = np.loadtxt(self.root_dir / 'annotations.csv', delimiter=',', dtype=np.int64)

    def __len__(self):
        return len(self.music)

    def __getitem__(self, idx):
        x = self.music[idx]
        y = self.labels[idx]

        if self.is_train:
            annotation = self.annotations[idx]
            return x, y, annotation
        else:
            return x, y
