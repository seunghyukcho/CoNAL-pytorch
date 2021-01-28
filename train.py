import torch
from torch.utils.data import DataLoader

from dataset import LabelMeDataset
from model import CoNAL
from classifier import LabelMeClassifier


if __name__ == "__main__":
    train_dataset = LabelMeDataset('data/LabelMe/train', 'data/LabelMe/images', True)
    classifier = LabelMeClassifier()
    model = CoNAL(3 * 256 * 256, 8, train_dataset.annotations.shape[1], classifier).double()

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    for epoch in range(1):
        for x, y, annotation in train_loader:
            annotator = torch.eye(77)
            result = model(x, annotator)

            print(result.shape)
