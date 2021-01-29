import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils import data


def add_dataset_args(parser):
    group = parser.add_argument_group('dataset')


class Dataset(data.Dataset):
    def __init__(self, args, root_dir, is_train=False, transforms=None):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / 'images'
        self.is_train = is_train
        self.transforms = transforms

        self.labels = np.loadtxt(self.root_dir / 'labels.txt', dtype=np.float32)
        with open(self.root_dir / 'filenames.txt', 'r') as f:
            self.image_paths = f.read().splitlines()
        if self.is_train:
            self.annotations = np.loadtxt(self.root_dir / 'annotations.txt', dtype=np.int64)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_dir / self.image_paths[idx])
        x = np.asarray(img)
        x = x / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = x.astype(np.float32)
        y = self.labels[idx]

        if self.is_train:
            annotation = self.annotations[idx]
            return x, y, annotation
        else:
            return x, y
