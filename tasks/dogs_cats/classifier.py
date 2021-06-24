import torch.nn as nn


def add_classifier_args(parser):
    group = parser.add_argument_group('classifier')
    group.add_argument('--dropout', type=float, default=0.5,
                       help="Dropout rate (Default: 0.5).")
    group.add_argument('--n_units', type=int, default=128,
                       help="Number of units in FC layer (Default: 128).")


class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_dim = args.input_dim

        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d(2)
        self.bn = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3))
        self.pool4 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pool1(x)
        x = self.bn(x)
        x = self.activation(self.conv2(x))
        x = self.pool2(x)
        x = self.activation(self.conv3(x))
        x = self.pool3(x)
        x = self.activation(self.conv4(x))
        x = self.pool4(x)

        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

