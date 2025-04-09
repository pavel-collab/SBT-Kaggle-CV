import torch
from models.dataframe import CustomDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.classification_head import (ClassificationHead1, 
                                        ClassificationHead2, 
                                        ClassificationHead3, 
                                        ClassificationHead4, 
                                        ClassificationHead5)
from models.model import (CustomResNet, 
                          CustomResNet50, 
                          CustomAlexNet, 
                          CustomGoogLeNet, 
                          CustomMobileNetV3, 
                          CustomResNet101, 
                          CustomMobileNetV3Large, 
                          CustomConvNeXtTiny, 
                          CustomEfficientNetB0)

#===========================================================================================================================#
''' Paths to data test, validate, train '''
train_csv_file = './data/train/train.csv'
train_images_dir = './data/train/images'

validation_csv_file = './data/validation/validation.csv'
validation_images_dir = './data/validation/images'

test_csv_file = './data/test.csv'
test_images_dir = './data/images'
#===========================================================================================================================#

#===========================================================================================================================#
''' Default ml pipeline preferences '''
batch_size = 64
num_epochs = 45
learning_rate = 0.0001
#===========================================================================================================================#

#===========================================================================================================================#
''' classes parameters '''
classes_list = ['healthy', 'multiple_diseases', 'rust', 'scab']
n_classes = len(classes_list)
class_weights = torch.tensor([0.8654891304347826, 5.137096774193548, 0.7288329519450801, 0.7825552825552825])
#===========================================================================================================================#

#===========================================================================================================================#
''' lists of models and heads of classification '''
models = {
    "resnet": CustomResNet,
    "resnet50": CustomResNet50,
    "alexnet": CustomAlexNet,
    "googlenet": CustomGoogLeNet,
    "mobilenet_v3": CustomMobileNetV3,
    "resnet101": CustomResNet101,
    "mobilenet_large": CustomMobileNetV3Large,
    "convnexttiny": CustomConvNeXtTiny,
    "efficientnetb0": CustomEfficientNetB0
}

classification_heads = {
    # "head_1": ClassificationHead1,
    "head_2": ClassificationHead2,
    "head_3": ClassificationHead3,
    "head_4": ClassificationHead4,
    "head_5": ClassificationHead5,
}
#===========================================================================================================================#

''' Datasets and dataloaders '''
#===========================================================================================================================#
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomRotation(15),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(0.5),  # Добавим вертикальное отражение
    transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # специальные значение нормализации для resnet
])

train_dataset = CustomDataset(train_csv_file, 
                              train_images_dir, 
                              classes_list, 
                              train_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
#===========================================================================================================================#

#===========================================================================================================================#
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # специальные значение нормализации для resnet
])

val_dataset = CustomDataset(validation_csv_file, 
                            validation_images_dir, 
                            classes_list, 
                            val_transform)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
#===========================================================================================================================#

#===========================================================================================================================#
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # специальные значение нормализации для resnet
])

test_dataset = CustomDataset(test_csv_file, 
                             test_images_dir, 
                             classes_list, 
                             test_transform, # should be commented to print image
                             is_test=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
#===========================================================================================================================#