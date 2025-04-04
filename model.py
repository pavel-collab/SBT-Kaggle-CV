import torch.nn as nn
import torchvision.models as models

class CustomAlexNet(nn.Module):
    def __init__(self, classification_head, n_classes: int, pretrained=True):
        super(CustomAlexNet, self).__init__()
        
        self.backbone = models.alexnet(pretrained=pretrained)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = classification_head(in_features, n_classes)
    
    def forward(self, x):
        return self.backbone(x)

class CustomGoogLeNet(nn.Module):
    def __init__(self, classification_head, n_classes: int, pretrained=True):
        super(CustomGoogLeNet, self).__init__()
        
        self.backbone = models.googlenet(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = classification_head(in_features, n_classes)
    
    def forward(self, x):
        return self.backbone(x)

class CustomResNet(nn.Module):
    def __init__(self, classification_head, n_classes: int, pretrained=True):
        super(CustomResNet, self).__init__()

        self.backbone = models.resnet18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = classification_head(in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)
    
class CustomResNet50(nn.Module):
    def __init__(self, classification_head, n_classes: int, pretrained=True):
        super(CustomResNet50, self).__init__()

        self.backbone = models.resnet50(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = classification_head(in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)

class CustomMobileNetV3(nn.Module):
    def __init__(self, classification_head, n_classes: int, pretrained=True):
        super(CustomMobileNetV3, self).__init__()

        self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = classification_head(in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)
