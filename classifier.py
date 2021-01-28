import torch.nn as nn
import torchvision.models as models


class LabelMe(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = models.vgg16(pretrained=True)
        self.backbone.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 8),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.backbone(x)
        return x
