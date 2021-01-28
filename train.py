import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import LabelMeDataset
from model import CoNAL
from classifier import LabelMeClassifier


if __name__ == "__main__":
    train_dataset = LabelMeDataset('data/LabelMe/train', 'data/LabelMe/images', True)
    classifier = LabelMeClassifier()
    model = CoNAL(3 * 256 * 256, 8, train_dataset.annotations.shape[1], classifier).cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = Adam(model.parameters(), lr=1e-3)

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    for epoch in range(10):
        for x, y, annotation in train_loader:
            model.zero_grad()
            x, y, annotation = x.cuda(), y.cuda(), annotation.cuda()
            annotator = torch.eye(77).cuda()
            result = model(x, annotator)

            result = torch.reshape(result, (-1, 8))
            annotation = annotation.view(-1)

            loss = criterion(result, annotation)
            loss.backward()
            optimizer.step()
            print(loss.item())
