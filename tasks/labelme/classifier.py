import torch.nn as nn
import torchvision.models as models


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

        self.backbone = models.vgg16_bn(pretrained=True)
        self.backbone.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, self.n_units),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.n_units, self.n_class),
            nn.Softmax(dim=-1)
        )

        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        return x
