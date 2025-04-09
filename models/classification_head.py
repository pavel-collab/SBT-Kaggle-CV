import torch.nn as nn

class ClassificationHead1(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(ClassificationHead1, self).__init__()
        
        self.classification_head = nn.Sequential(
            nn.Linear(in_features, out_features)
        )
        
    def forward(self, x):
        return self.classification_head(x)
    
class ClassificationHead2(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(ClassificationHead2, self).__init__()
        
        self.classification_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, out_features)
        )
        
    def forward(self, x):
        return self.classification_head(x)
    
class ClassificationHead3(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(ClassificationHead3, self).__init__()
        
        self.classification_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, out_features)
        )
        
    def forward(self, x):
        return self.classification_head(x)
    
class ClassificationHead4(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(ClassificationHead4, self).__init__()
        
        self.classification_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, out_features)
        )
        
    def forward(self, x):
        return self.classification_head(x)
    
class ClassificationHead5(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(ClassificationHead5, self).__init__()
        
        self.classification_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, out_features)
        )
        
    def forward(self, x):
        return self.classification_head(x)