import argparse
from pathlib import Path
import torch
from models.dataframe import CustomDataset
import torchvision.transforms as transforms
from utils.utils import evaluate_model, plot_confusion_matrix, extract_model_name, get_device
from torch.utils.data import DataLoader
from utils.constants import (models,
                             classification_heads,
                             val_loader,
                             classes_list)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', help='set a path to the model that we want to evaluate')
parser.add_argument('-o', '--output', help='set a path to the output filename, programm will write a model name and final accuracy')
args = parser.parse_args()

if args.output is not None and args.output != "":
    output_file_path = Path(args.output)
else: 
    output_file_path = None
    
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
device = get_device()

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