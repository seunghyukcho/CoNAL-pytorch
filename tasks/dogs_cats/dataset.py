import numpy as np
from pathlib import Path
from torch.utils import data
from torchvision import transforms


def add_dataset_args(parser):
    group = parser.add_argument_group('dataset')


class Dataset(data.Dataset):
    def __init__(self, args, root_dir, is_train=False):
        self.root_dir = Path(root_dir)
        self.is_train = is_train

        self.images = np.load(self.root_dir / 'data.npy')
        # self.images = np.transpose(self.images, (0, 3, 1, 2))

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
            ]) 

        self.labels = np.load(self.root_dir / 'labels.npy')
        if self.is_train:
            self.annotations = np.load(self.root_dir / 'annotations.npy')

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        x = self.images[idx]
        x = self.transforms(x)
        y = self.labels[idx]

        if self.is_train:
            annotation = self.annotations[idx]
            return x, y, annotation
        else:
            return x, y

