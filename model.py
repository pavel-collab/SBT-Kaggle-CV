import torch.nn as nn
import torchvision.models as models

class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(ClassificationHead, self).__init__()
        
        self.classification_head = nn.Sequential(
            nn.Linear(in_features, out_features)
        )
        
    def forward(self, x):
        return self.classification_head(x)

class CustomAlexNet(nn.Module):
    def __init__(self, n_classes: int, pretrained=True):
        super(CustomAlexNet, self).__init__()
        
        self.backbone = models.alexnet(pretrained=pretrained)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = ClassificationHead(in_features, n_classes)
    
    def forward(self, x):
        return self.backbone(x)

class CustomGoogLeNet(nn.Module):
    def __init__(self, n_classes: int, pretrained=True):
        super(CustomGoogLeNet, self).__init__()
        
        self.backbone = models.googlenet(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = ClassificationHead(in_features, n_classes)
    
    def forward(self, x):
        return self.backbone(x)

class CustomResNet(nn.Module):
    def __init__(self, n_classes: int, pretrained=True):
        super(CustomResNet, self).__init__()

        self.backbone = models.resnet18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = ClassificationHead(in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)
    
class CustomConvNeXt(nn.Module):
    def __init__(self, n_classes: int, pretrained=True):
        super(CustomConvNeXt, self).__init__()

        self.backbone = models.convnext_base(pretrained=pretrained)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = ClassificationHead(in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)
    
class CustomEfficientNet(nn.Module):
    def __init__(self, n_classes: int, pretrained=True):
        super(CustomEfficientNet, self).__init__()

        self.backbone = models.efficientnet_b1(pretrained=pretrained)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = ClassificationHead(in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)

class CustomMobileNetV3(nn.Module):
    def __init__(self, n_classes: int, pretrained=True):
        super(CustomMobileNetV3, self).__init__()

        self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = ClassificationHead(in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)
