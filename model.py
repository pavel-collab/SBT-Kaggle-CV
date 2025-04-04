import torch.nn as nn
import torchvision.models as models

class CustomAlexNet(nn.Module):
    def __init__(self, n_classes: int, classification_head=None, pretrained=True):
        super(CustomAlexNet, self).__init__()
        
        self.backbone = models.alexnet(pretrained=pretrained)
        in_features = self.backbone.classifier[-1].in_features
        
        self.classification_head = classification_head
        if self.classification_head is None:
            self.backbone.classifier[-1] = self.classification_head = nn.Sequential(
                nn.Linear(in_features, n_classes)
            )
        else:
            self.backbone.classifier[-1] = classification_head(in_features, n_classes)
    
    def forward(self, x):
        return self.backbone(x)

class CustomGoogLeNet(nn.Module):
    def __init__(self, n_classes: int, classification_head=None, pretrained=True):
        super(CustomGoogLeNet, self).__init__()
        
        self.backbone = models.googlenet(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        
        self.classification_head = classification_head
        if self.classification_head is None:
            self.backbone.fc = nn.Sequential(
                nn.Linear(in_features, n_classes)
            )    
        else:
            self.backbone.fc = classification_head(in_features, n_classes)
    
    def forward(self, x):
        return self.backbone(x)

class CustomResNet(nn.Module):
    def __init__(self, n_classes: int, classification_head=None, pretrained=True):
        super(CustomResNet, self).__init__()

        self.backbone = models.resnet18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        
        self.classification_head = classification_head
        if self.classification_head is None:
            self.backbone.fc = nn.Sequential(
                nn.Linear(in_features, n_classes)
            )    
        else:
            self.backbone.fc = classification_head(in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)
    
class CustomResNet50(nn.Module):
    def __init__(self, n_classes: int, classification_head=None, pretrained=True):
        super(CustomResNet50, self).__init__()

        self.backbone = models.resnet50(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        
        self.classification_head = classification_head
        if self.classification_head is None:
            self.backbone.fc = nn.Sequential(
                nn.Linear(in_features, n_classes)
            )    
        else:
            self.backbone.fc = classification_head(in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)

class CustomResNet101(nn.Module):
    def __init__(self, n_classes: int, classification_head=None, pretrained=True):
        super(CustomResNet101, self).__init__()

        self.backbone = models.resnet101(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        
        self.classification_head = classification_head
        if self.classification_head is None:
            self.backbone.fc = nn.Sequential(
                nn.Linear(in_features, n_classes)
            )    
        else:
            self.backbone.fc = classification_head(in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)

class CustomMobileNetV3(nn.Module):
    def __init__(self, n_classes: int, classification_head=None, pretrained=True):
        super(CustomMobileNetV3, self).__init__()

        self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
        in_features = self.backbone.classifier[-1].in_features
        
        self.classification_head = classification_head
        if self.classification_head is None:
            self.backbone.classifier[-1] = self.classification_head = nn.Sequential(
                nn.Linear(in_features, n_classes)
            )
        else:
            self.backbone.classifier[-1] = classification_head(in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)
    
class CustomMobileNetV3Large(nn.Module):
    def __init__(self, n_classes: int, classification_head=None, pretrained=True):
        super(CustomMobileNetV3Large, self).__init__()

        self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
        in_features = self.backbone.classifier[-1].in_features
        
        self.classification_head = classification_head
        if self.classification_head is None:
            self.backbone.classifier[-1] = self.classification_head = nn.Sequential(
                nn.Linear(in_features, n_classes)
            )
        else:
            self.backbone.classifier[-1] = classification_head(in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)

class CustomConvNeXtSmall(nn.Module):
    def __init__(self, n_classes: int, classification_head=None, pretrained=True):
        super(CustomConvNeXtSmall, self).__init__()

        self.backbone = models.convnext_small(pretrained=pretrained)
        in_features = self.backbone.classifier[-1].in_features
        
        self.classification_head = classification_head
        if self.classification_head is None:
            self.backbone.classifier[-1] = self.classification_head = nn.Sequential(
                nn.Linear(in_features, n_classes)
            )
        else:
            self.backbone.classifier[-1] = classification_head(in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)
    
class CustomConvNeXtTiny(nn.Module):
    def __init__(self, n_classes: int, classification_head=None, pretrained=True):
        super(CustomConvNeXtTiny, self).__init__()

        self.backbone = models.convnext_tiny(pretrained=pretrained)
        in_features = self.backbone.classifier[-1].in_features
        
        self.classification_head = classification_head
        if self.classification_head is None:
            self.backbone.classifier[-1] = self.classification_head = nn.Sequential(
                nn.Linear(in_features, n_classes)
            )
        else:
            self.backbone.classifier[-1] = classification_head(in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)

class CustomEfficientNetB0(nn.Module):
    def __init__(self, n_classes: int, classification_head=None, pretrained=True):
        super(CustomEfficientNetB0, self).__init__()

        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        in_features = self.backbone.classifier[-1].in_features
        
        self.classification_head = classification_head
        if self.classification_head is None:
            self.backbone.classifier[-1] = self.classification_head = nn.Sequential(
                nn.Linear(in_features, n_classes)
            )
        else:
            self.backbone.classifier[-1] = classification_head(in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)