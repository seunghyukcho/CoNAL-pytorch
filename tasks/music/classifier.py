import torch.nn as nn


def add_classifier_args(parser):
    group = parser.add_argument_group('classifier')
    group.add_argument('--dropout', type=float, default=0.5,
                       help="Dropout rate")
    group.add_argument('--n_units', type=int, default=128,
                       help="Number of units in FC layer")


class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.n_units = args.n_units
        self.n_class = args.n_class

        self.layers = nn.Sequential(
            nn.Linear(124, self.n_units),
            nn.BatchNorm1d(self.n_units, affine=False),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.n_units, self.n_class),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
