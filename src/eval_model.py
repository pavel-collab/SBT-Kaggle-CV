import argparse
from pathlib import Path
import torch
<<<<<<< HEAD
from models.dataframe import CustomDataset
import torchvision.transforms as transforms
from utils.utils import evaluate_model, plot_confusion_matrix, extract_model_name, get_device
from torch.utils.data import DataLoader
from utils.constants import (models,
                             classification_heads,
                             val_loader,
                             classes_list)
=======
from dataframe import CustomDataset
import torchvision.transforms as transforms
from utils import evaluate_model, plot_confusion_matrix
from torch.utils.data import DataLoader
from model import CustomResNet, CustomResNet50, CustomAlexNet, CustomGoogLeNet, CustomMobileNetV3, CustomResNet101, CustomMobileNetV3Large, CustomConvNeXtTiny, CustomEfficientNetB0
from classification_head import ClassificationHead1, ClassificationHead2, ClassificationHead3, ClassificationHead4, ClassificationHead5

import cv2

from albumentations.pytorch import ToTensorV2
from albumentations import (
    Compose,
    Resize,
    OneOf,
    RandomBrightnessContrast,
    MotionBlur,
    MedianBlur,
    GaussianBlur,
    VerticalFlip,
    HorizontalFlip,
    ShiftScaleRotate,
    Normalize,

)

#TODO: we can see this constants in different files here, we can move it to the separate file and import in each file, where we need it
classes_list = ['healthy', 'multiple_diseases', 'rust', 'scab']
batch_size = 64
validation_csv_file = './data/validation/validation.csv'
validation_images_dir = './data/validation/images'

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
>>>>>>> 227806b (add directories)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', help='set a path to the model that we want to evaluate')
parser.add_argument('-o', '--output', help='set a path to the output filename, programm will write a model name and final accuracy')
args = parser.parse_args()

if args.output is not None and args.output != "":
    output_file_path = Path(args.output)
else: 
    output_file_path = None
<<<<<<< HEAD
=======

def extract_model_name(model_file_name: str):
    if 'best_model' in model_file_name:
        return model_file_name.removeprefix("best_model_").removesuffix(".pth")
    elif 'last_model' in model_file_name:
        return model_file_name.removeprefix("last_model_").removesuffix(".pth")
    else:
        return None
>>>>>>> 227806b (add directories)
    
model_file_path = Path(args.model_path)
assert(model_file_path.exists())

model_name = extract_model_name(model_file_path.name)
assert(model_name is not None)
assert(model_name in models.keys())

# check if we use custom classification head
if 'head' in model_file_path.parent.name:
    head_name = model_file_path.parent.name
else:
    head_name = None

# детектируем девайс
<<<<<<< HEAD
device = get_device()
=======
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

val_transform = Compose([
                    Resize(height=HEIGHT, width=WIDTH),
                    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
                    ToTensorV2()
                ])

val_dataset = CustomDataset(validation_csv_file, 
                            validation_images_dir, 
                            classes_list, 
                            val_transform)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
>>>>>>> 227806b (add directories)

if head_name is None:
    model = models[model_name](n_classes=len(classes_list))
else:
    model = models[model_name](n_classes=len(classes_list), classification_head=classification_heads[head_name])
assert(model is not None)

model.load_state_dict(torch.load(model_file_path.absolute()))
model.to(device)

try:
    # Оценка модели и построение матрицы ошибок
    cm, report, accuracy_1, weighted_f1_1 = evaluate_model(model, val_loader, device)
    print(f"Metrics for model {model_name}:")
    print(report)
    print(f'Test Accuracy: {accuracy_1:.4f}')
    plot_confusion_matrix(cm, classes=range(len(classes_list)), model_name=model_name, save_file_path='./images/')
    
    if output_file_path is not None:
        file_create = output_file_path.exists()
        
        with open(output_file_path.absolute(), 'a') as fd:
            if not file_create:
                fd.write("model,accuracy\n")
            fd.write(f"{model_name}_{head_name},{accuracy_1}\n")
except Exception as ex:
    print(f"During evaluating model {model_name} we have faced with exception {ex}")