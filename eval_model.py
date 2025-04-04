import argparse
from pathlib import Path
import torch
from dataframe import CustomDataset
import torchvision.transforms as transforms
from utils import evaluate_model, plot_confusion_matrix
from torch.utils.data import DataLoader
from model import CustomResNet, CustomAlexNet, CustomGoogLeNet, CustomMobileNetV3, CustomResNet50

#TODO: we can see this constants in different files here, we can move it to the separate file and import in each file, where we need it
classes_list = ['healthy', 'multiple_diseases', 'rust', 'scab']
batch_size = 64
validation_csv_file = './data/validation/validation.csv'
validation_images_dir = './data/validation/images'

models = {
    "resnet": CustomResNet,
    "alexnet": CustomAlexNet,
    "googlenet": CustomGoogLeNet,
    "mobilenet_v3": CustomMobileNetV3,
    "resnet50": CustomResNet50
}

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', help='set a path to the model that we want to evaluate')
args = parser.parse_args()

def extract_model_name(model_file_name: str):
    if 'best_model' in model_file_name:
        return model_file_name.removeprefix("best_model_").removesuffix(".pth")
    elif 'last_model' in model_file_name:
        return model_file_name.removeprefix("last_model_").removesuffix(".pth")
    else:
        return None
    
model_file_path = Path(args.model_path)
assert(model_file_path.exists())

model_name = extract_model_name(model_file_path.name)
assert(model_name is not None)
assert(model_name in models.keys())

# детектируем девайс
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = models[model_name](n_classes=len(classes_list))
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
except Exception as ex:
    print(f"During evaluating model {model_name} we have faced with exception {ex}")