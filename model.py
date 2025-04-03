import torch.nn as nn
import torchvision.models as models

class CVModel(nn.Module):
    def __init__(self, n_classes: int, pretrained=True):
        super(CVModel, self).__init__()

        self.backbone = models.resnet18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, n_classes)
        )

    def forward(self, x):
        return self.backbone(x)
