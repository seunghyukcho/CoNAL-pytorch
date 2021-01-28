import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


class LabelMeDataset(Dataset):
    def __init__(self, root_dir, image_dir, is_train=False):
        self.root_dir = Path(root_dir)
        self.image_dir = Path(image_dir)
        self.is_train = is_train

        self.labels = np.loadtxt(self.root_dir / 'labels.txt')
        with open(self.root_dir / 'filenames.txt', 'r') as f:
            self.image_paths = f.read().splitlines()
        if self.is_train:
            self.annotations = np.loadtxt(self.root_dir / 'annotations.txt')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_dir / self.image_paths[idx])
        x = np.asarray(img)
        x = x / 255.0
        x = np.transpose(x, (2, 0, 1))
        y = self.labels[idx]

        if self.is_train:
            annotation = self.annotations[idx]
            return x, y, annotation
        else:
            return x, y
