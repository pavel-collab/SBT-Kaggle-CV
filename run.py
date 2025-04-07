import torch 
import numpy as np
import argparse
from itertools import dropwhile
from pathlib import Path

from dataframe import CustomDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import logging

from classification_head import ClassificationHead1, ClassificationHead2, ClassificationHead3, ClassificationHead4, ClassificationHead5
from model import CustomResNet, CustomResNet50, CustomAlexNet, CustomGoogLeNet, CustomMobileNetV3, CustomResNet101, CustomMobileNetV3Large, CustomConvNeXtTiny, CustomEfficientNetB0
from utils import train_model, plot_train_proces, last_model_settings
import torch.profiler as profiler

import logger_config
from logger_config import DEFAULT_LOG_DIRECTORY

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--resume', action='store_true', help='resume mode')
args = parser.parse_args()

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

class_weights = torch.tensor([0.8654891304347826, 5.137096774193548, 0.7288329519450801, 0.7825552825552825])

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

# число эпох
num_epochs = 45
learning_rate = 0.0001

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
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
if args.resume:
    logfile_path = Path(DEFAULT_LOG_DIRECTORY)
    assert(logfile_path.exists())

    last_model_name, last_head_name = last_model_settings(logfile_path.absolute())
    logger.info(f"Resume model training from {last_model_name} and {last_head_name}")
else:
    last_model_name, last_head_name = models.keys()[0], classification_heads.keys()[0]

try:
    with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/profiler'),
    ) as prof:
        for model_name, model_class in dropwhile(lambda item: item[0] < last_model_name, models.items()):
            for class_head_name, cl_head in dropwhile(lambda item: item[0] < last_head_name, classification_heads.items()):
                logger.info(f"Start to train model {model_name} with classification head {class_head_name}")
                model = model_class(classification_head=cl_head, n_classes=n_classes)
                try:
                    model_train_result = train_model(model,
                                                    model_name,
                                                    device,
                                                    num_epochs,
                                                    learning_rate,
                                                    train_loader,
                                                    val_loader,
                                                    classification_head_name=class_head_name,
                                                    class_weights=class_weights
                                                    )
                    plot_train_proces(num_epochs,
                                    model_train_result.train_losses,
                                    model_train_result.val_losses,
                                    model_train_result.train_accuracies,
                                    model_train_result.val_accuracies,
                                    model_name,
                                    classification_head_name=class_head_name)
                except Exception as ex:
                    logger.error(f"During training model {model_name} was caught exception {ex}")
                    continue
except KeyboardInterrupt:
    print("The program was interrupted by key signal")