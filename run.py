import torch 
import numpy as np

from dataframe import CustomDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import logging

from model import CustomResNet, CustomAlexNet, CustomGoogLeNet, CustomConvNeXt, CustomEfficientNet, CustomMobileNetV3
from utils import train_model, plot_train_proces, TrainModelResult

import logger_config

logger = logging.getLogger(__name__)

# фиксируем рандомный сид
seed  = 20
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False

# детектируем девайс
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes_list = ['healthy', 'multiple_diseases', 'rust', 'scab']

train_csv_file = './data/train/train.csv'
train_images_dir = './data/train/images'
validation_csv_file = './data/validation/validation.csv'
validation_images_dir = './data/validation/images'

batch_size = 64

n_classes = len(classes_list)

models = {
    "resnet": CustomResNet,
    "alexnet": CustomAlexNet,
    "googlenet": CustomGoogLeNet,
    "convnext": CustomConvNeXt,
    "efficientnet": CustomEfficientNet,
    "mobilenet_v3": CustomMobileNetV3
}

# число эпох
num_epochs = 70
learning_rate = 0.0005

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

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomRotation(10),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(0.5),  # Добавим вертикальное отражение
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # специальные значение нормализации для resnet
])

train_dataset = CustomDataset(train_csv_file, 
                              train_images_dir, 
                              classes_list, 
                              train_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
for model_name, model_class in models.items():
    model = model_class(n_classes=n_classes)
    try:
        model_train_result = train_model(model,
                                         model_name,
                                         device,
                                         num_epochs,
                                         learning_rate,
                                         train_loader,
                                         val_loader)
        plot_train_proces(num_epochs,
                          model_train_result.train_losses,
                          model_train_result.val_losses,
                          model_train_result.train_accuracies,
                          model_train_result.val_accuracies,
                          model_name)
    except Exception as ex:
        logger.error(f"During training model {model_name} was caught exception {ex}")
        continue